# swiglu_group_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T15:47:24.121Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description:

  Performs grouped low-bit quantization after applying the SwiGLU activation to input `x`. Supports Block FP8, MX FP8, HiFloat8 static quantization, and HiFloat8 dynamic quantization. The bottom layer encapsulates `aclnnSwigluGroupQuant`.

- Calculation Process:

  1. Splits the last dimension of `x` evenly into `A` and `B`, and computes the SwiGLU result `silu(A) * B`.
  2. When `weight` is not empty, multiplies the SwiGLU result by `weight` token by token to obtain the quantization input.
  3. Quantize the weighted result according to `quant_mode` and output `y` and `y_scale`.
  4. When `output_origin=True`, additionally return the pre-weighted SwiGLU result `y_origin`; otherwise, return a placeholder tensor of shape `[0]`.

## Function Prototype

```python
cann_ops_nn.swiglu_group_quant(
    x,
    *,
    weight=None,
    group_index=None,
    scale=None,
    dst_type=291,
    quant_mode=0,
    block_size=0,
    round_scale=False,
    clamp_limit=-1.0,
    dst_type_max=15.0,
    output_origin=False,
) -> (Tensor, Tensor, Tensor)
```

## Parameters

| Name | Type | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- | --- |
| `x` | Tensor | Required | SwiGLU input. The last dimension is evenly split into two parts. | `torch.float16`, `torch.bfloat16` (extra support for `torch.float32` when quant_mode is 2/3) | 2-8 dimensions (2-7 dimensions when quant_mode is 1) |
| `weight` | Tensor | Optional | Token-wise weight. If not empty, it is multiplied with the SwiGLU result, and the weighted result is used for quantization. | `torch.float32` | 1-8 dimensions, with the number of elements equal to the number of elements of `x` excluding the last dimension |
| `group_index` | Tensor | Optional | Number of grouped tokens in count mode. | `torch.int64` | 1 dimension |
| `scale` | Tensor | Optional | Scale used in HiFloat8 static quantization. Used only when quant mode is 2. | `torch.float32` | 1 dimension |
| `dst_type` | int | Optional | torch dtype encoding of the target quantization type. The default value is `291`. | - | - |
| `quant_mode` | int | Optional | Quantization mode. Supported values are `0`, `1`, `2`, and `3`. | - | - |
| `block_size` | int | Optional | Quantization block size. `0` indicates that the mode default value is used. | - | - |
| `round_scale` | bool | Optional | Whether the scale is rounded to a power of 2 in MX quantization. | - | - |
| `clamp_limit` | float | Optional | Clamping threshold before the SwiGLU computation. The default value `-1.0` indicates that clamping is disabled. | - | - |
| `dst_type_max` | float | Optional | Maximum finite value used to compute the scale in HiFloat8 dynamic quantization. | - | - |
| `output_origin` | bool | Optional | Whether to return the pre-weighted SwiGLU result. | - | - |

### quant_mode and dst_type

| `quant_mode` | Meaning | Supported Values of `dst_type` | Torch Dtype of `y` | Torch Dtype of `y_scale` |
| --- | --- | --- | --- | --- |
| `0` | Block FP8 | `23`/`291` indicates `float8_e5m2`; `24`/`292` indicates `float8_e4m3fn`. | `torch.float8_e5m2` or `torch.float8_e4m3fn` | `torch.float32` |
| `1` | MX FP8 | `23`, `24`, `291`, `292` | `torch.float8_e5m2` or `torch.float8_e4m3fn` | `torch.float8_e8m0fnu` |
| `2` | HiFloat8 static quantization | `290` indicates HiFloat8. | `torch.uint8` | `torch.float32`, with shape `[0]` |
| `3` | HiFloat8 dynamic quantization | `290` indicates HiFloat8. | `torch.uint8` | `torch.float32` |

#### dst_type Encoding Description

| `dst_type` | Corresponding Type | Source | Type Dispatched To GE/ACL | Applicable `quant_mode` |
| --- | --- | --- | --- | --- |
| `23` | `torch.float8_e5m2` | Native PyTorch dtype int value | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`, `1` |
| `24` | `torch.float8_e4m3fn` | Native PyTorch dtype int value | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`, `1` |
| `291` | `torch_npu.float8_e5m2`, same semantics as `torch.float8_e5m2` | torch_npu extension dtype encoding | `DT_FLOAT8_E5M2` / `ACL_FLOAT8_E5M2` | `0`, `1` |
| `292` | `torch_npu.float8_e4m3fn`, same semantics as `torch.float8_e4m3fn` | torch_npu extension dtype encoding | `DT_FLOAT8_E4M3FN` / `ACL_FLOAT8_E4M3FN` | `0`, `1` |
| `290` | `torch_npu.hifloat8` | torch_npu extension dtype encoding | `DT_HIFLOAT8` / `ACL_HIFLOAT8` | `2`, `3` |

Note: `quant_mode=2/3` indicates HiFloat8 mode, and the actual dispatch is `DT_HIFLOAT8`/`ACL_HIFLOAT8`. graph_convert converts the torch dtype encodings in the preceding table to GE dtype attrs. The ACLNN path converts the corresponding encodings to `aclDataType` and then calls the underlying API.

## Return Values

| Name | Type | Description | Data Type | Dimension (Shape) |
| --- | --- | --- | --- | --- |
| `y` | Tensor | Quantized output. | See `quant_mode and dst_type` | `x.shape[:-1] + [D/2]` |
| `y_scale` | Tensor | Quantization scale output. | See `quant_mode and dst_type` | `x.shape[:-1] + [ceil((D/2)/128)]` when `quant_mode=0`; `x.shape[:-1] + [ceil(ceil((D/2)/32)/2), 2]` when `quant_mode=1`; `[0]` when `quant_mode=2`; `group_index.shape` or `[1]` when `quant_mode=3` |
| `y_origin` | Tensor | Pre-weighted SwiGLU result of weight or a placeholder Tensor. | Same as `x` | `x.shape[:-1] + [D/2]` when `output_origin=True`, otherwise `[0]` |

where `D = x.shape[-1]`.

## Constraints

- This API can be called in single-operator mode and TorchAir graph mode.
- `x`, `weight`, `group_index`, and `scale` must be NPU Tensors; optional Tensors can be set to `None`.
- Input `x` is 2-8D (2-7D when quant_mode is 1), and the last dimension `D` must be greater than or equal to 256 and divisible by 256.
- `dst_type` supports the torch dtype encodings of FP8 and HiFloat8. For details, see `dst_type encoding description`.
- When `quant_mode=0`, only FP8 output is supported. `dst_type` supports `23`, `24`, `291`, and `292`. `block_size` supports `0` or `128`.
- When `quant_mode=1`, FP8 output is supported. `dst_type` supports `23`, `24`, `291`, and `292`. `block_size` supports `0` or `32`. `round_scale` must be `True`.
- When `quant_mode=2`, HiFloat8 static quantization output is supported. `scale` is required. `dst_type`, `block_size`, and `round_scale` do not take effect, and the actual dispatch is HiFloat8.
- When `quant_mode=3`, HiFloat8 dynamic quantization output is supported, `scale` is not used, and `dst_type`, `block_size`, and `round_scale` do not take effect. HiFloat8 is actually dispatched.
- The data type of `y_scale` must match `quant_mode`: `torch.float32` for Block FP8, `torch.float8_e8m0fnu` for MX, and `torch.float32` for HiFloat8.
- When `quant_mode=3`, `group_index` can be used for group dynamic quantization in MoE use cases. The shape of `y_scale` is `group_index.shape`, or `[1]` when `group_index` is not passed.
- When `clamp_limit` is disabled, the default placeholder value `-1.0` is used; when enabled, it must be greater than 0.
- Element values in group_index must be greater than or equal to 0.
- Inputs `x` and `weight` support non-contiguous tensors. Non-contiguous `group_index` and `scale` are not supported.
- When quant_mode is 0 or 1, empty tensors are supported for x and weight, but group_index cannot be an empty tensor alone.

## Deterministic Computation

Defaults to a deterministic computation.

## API Call

- Single-operator mode call:

  ```python
  import torch
  import torch_npu
  import cann_ops_nn.ops

  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y, y_scale, y_origin = torch.ops.cann_ops_nn.swiglu_group_quant(
      x,
      dst_type=291,
      quant_mode=0,
      block_size=128,
  )
  ```

- Graph mode (torchair) call:

  ```python
  import torch
  import torch_npu
  import torchair
  import cann_ops_nn.ops

  class Model(torch.nn.Module):
      def forward(self, x):
          y, y_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
              x,
              dst_type=291,
              quant_mode=0,
              block_size=128,
          )
          return y, y_scale

  model = torch.compile(Model().npu(), backend=torchair.get_npu_backend(), dynamic=False)
  x = torch.randn(8, 512, dtype=torch.float16).npu()
  y, y_scale = model(x)
  ```
