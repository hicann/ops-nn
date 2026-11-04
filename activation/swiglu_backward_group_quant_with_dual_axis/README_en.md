# SwigluBackwardGroupQuantWithDualAxis

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

### Description

SwigluBackwardGroupQuantWithDualAxis fuses the SwiGLU backward computation with optional Clamp and weight, and the dynamic MX quantization along the last dimension and the second-to-last dimension. Currently, only `quantMode=1` is supported, and the operator outputs FP8 data and E8M0 scaling factors.

- Non-group scenario: do not pass `group_index`, `weight`, or `y_origin`; the operator returns `y1`, `scale1`, `y2`, and `scale2`.
- Group scenario: pass `group_index` in cumsum mode, and optionally pass `weight` and `y_origin` as a pair; when both are passed, `grad_weight` is returned additionally.
- The operator supports only 2-dim inputs: the shape of `x` is `[T, 2H]`, and the shape of `gradY` is `[T, H]`.

### Formula

Split `x` along the last dimension into `a` and `b`:

$$
a = x[:, :H], \qquad b = x[:, H:2H]
$$

When `clamp_limit > 0`:

$$
a_c = \min(a, \mathrm{clamp\_limit}), \qquad
b_c = \min(\max(b, -\mathrm{clamp\_limit}), \mathrm{clamp\_limit})
$$

Let $s=\sigma(\alpha a_c)$. When `weight` is not passed, let $g=grad_y$; when `weight` is passed, let $g=grad_y\times weight[:,None]$:

$$
grad_a = g \times (b_c+bias) \times s \times (1+\alpha a_c(1-s)) \times mask_a
$$

$$
grad_b = g \times a_c \times s \times mask_b
$$

Here, when Clamp is not enabled, both `mask_a` and `mask_b` are 1; when Clamp is enabled, they indicate that the corresponding inputs are not truncated respectively. The final `grad_x` is the concatenation of `grad_a` and `grad_b` along the last dimension. Computed only when `weight` and `y_origin` are passed together:

$$
grad\_weight[t] = \sum_{h=0}^{H-1} grad_y[t,h] \times y_origin[t,h]
$$

Quantization uses 32 elements as one MX block, and the scale is rounded to a power of two:

$$
scale = \mathrm{ceil\_power\_of\_two}\left(\frac{amax}{fp8\_max}\right),
\qquad output = \mathrm{cast}_{dst\_type}\left(\frac{grad_x}{scale}\right)
$$

`dst_type=35` uses FLOAT8_E5M2, and `dst_type=36` uses FLOAT8_E4M3FN. `scale1` is the quantization scaling factor along the last dimension, and `scale2` is the quantization scaling factor along the second-to-last dimension. In the group scenario, `scale2` uses a layout with group boundary offsets, where the extra rows carry group boundaries and are not ordinary invalid padding.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| gradY | Input | Backward gradient of the SwiGLU output, with shape `[T, H]`. | FLOAT16, BFLOAT16 | ND |
| x | Input | SwiGLU forward input, with shape `[T, 2H]`. | FLOAT16, BFLOAT16 | ND |
| weightOptional | Optional input | Weight of each token, with shape `[T]`; supported only in the group scenario. | FLOAT16, BFLOAT16, FLOAT | ND |
| yOriginOptional | Optional input | Original value of the SwiGLU forward output, with shape `[T, H]`; passed together with weight as a pair. | FLOAT16, BFLOAT16 | ND |
| groupIndexOptional | Optional input | Group accumulation endpoint, with shape `[G]`, in cumsum mode. | INT64 | ND |
| clampLimit | Attribute | Clamp threshold; `-1.0` means disabled, and the value must be greater than 0 when enabled. | DOUBLE | - |
| alpha | Attribute | Alpha coefficient in the SwiGLU backward computation, must be greater than 0. | DOUBLE | - |
| bias | Attribute | Bias coefficient in the SwiGLU backward computation. | DOUBLE | - |
| quantMode | Attribute | Quantization mode, only `1` is supported. | INT64 | - |
| dstType | Attribute | FP8 type: `35` is FLOAT8_E5M2, and `36` is FLOAT8_E4M3FN. | INT64 | - |
| y1Out | Output | Quantization result along the -1 axis, with the same shape as x. | FLOAT8_E4M3FN, FLOAT8_E5M2 | ND |
| scale1Out | Output | E8M0 scaling factor along the -1 axis, with shape `[T, ceil(2H/64), 2]`. | FLOAT8_E8M0 | ND |
| y2Out | Output | Quantization result along the -2 axis, with the same shape as x. | FLOAT8_E4M3FN, FLOAT8_E5M2 | ND |
| scale2Out | Output | E8M0 scaling factor along the -2 axis; `[ceil(T/64), 2H, 2]` for non-group and `[floor(T/64)+G, 2H, 2]` for group. | FLOAT8_E8M0 | ND |
| gradWeightOutOptional | Optional output | Gradient of weight, used only together with weight and yOrigin as a pair, with the same shape and type as weight. | FLOAT16, BFLOAT16, FLOAT | ND |

## Constraints

- Only Ascend 950 is supported. The inputs `x` and `grad_y` must be 2-dim tensors, with shapes `[T, 2H]` and `[T, H]` respectively.
- The last dimension of `x` must be a multiple of 64, and the last dimension of `grad_y` must equal half of the last dimension of `x`.
- In the non-group scenario, do not pass `group_index`, `weight`, or `y_origin`; in the group scenario, `group_index` must be passed.
- `group_index` uses the cumsum mode, must be strictly increasing, and its last value must equal `T`.
- `weight` supports only the group scenario; when `weight` is passed, `y_origin` must be passed at the same time, and the shapes and data types of the two must satisfy the Parameters section.
- `quant_mode` supports only `1`; `dst_type` supports only `35` and `36`.
- `clamp_limit` supports only `-1.0` or a value greater than 0, and `alpha` must be greater than 0.
- Input tensors support the non-contiguous format, and the operator prototype handles it through `AutoContiguous`; the data format of inputs and outputs is ND.
- Undefined parameters such as `interleaved` and `dim` are not supported, and batch consistency configuration is not supported either.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| aclnn invocation | [test_aclnn_swiglu_backward_group_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_backward_group_quant_with_dual_axis.cpp) | Calls the operator through the [aclnnSwigluBackwardGroupQuantWithDualAxis](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md) interface. |
| Graph-mode invocation | - | Defines the operator graph node through the [operator IR](./op_graph/swiglu_backward_group_quant_with_dual_axis_proto.h). |
| PyTorch API | - | Calls the operator through the [swiglu_backward_group_quant_with_dual_axis](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md) interface. |

## References

- [aclnn API document](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md)
- [PyTorch API document](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md)
