# SwigluBackwardGroupQuantWithDualAxis

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swiglu_backward_group_quant_with_dual_axis)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    x     |
| <term>Atlas A2系列产品</term> |    x     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    x    |

## 功能说明

### 接口功能

SwigluBackwardGroupQuantWithDualAxis 融合带可选 Clamp 和 weight 的 SwiGLU 反向计算，以及沿最后一维和倒数第二维的动态 MX 量化。当前仅支持 `quant_mode=1`，输出 FP8 数据和 E8M0 缩放因子。

- 非 group 场景：不传入 `group_index`、`weight` 和 `y_origin`，返回 `y1`、`scale1`、`y2` 和 `scale2`。
- group 场景：传入 cumsum 模式的 `group_index`，可选成对传入 `weight` 和 `y_origin`；传入二者时额外返回 `grad_weight`。
- 算子仅支持二维输入：`x` 的 shape 为 `[T, 2H]`，`grad_y` 的 shape 为 `[T, H]`。

### 计算公式

将 `x` 沿最后一维切分为 `a` 和 `b`：

$$
a = x[:, :H], \qquad b = x[:, H:2H]
$$

当 `clamp_limit > 0` 时：

$$
a_c = \min(a, \mathrm{clamp\_limit}), \qquad
b_c = \min(\max(b, -\mathrm{clamp\_limit}), \mathrm{clamp\_limit})
$$

令 $s=\sigma(\alpha a_c)$。不传入 `weight` 时令 $g=grad_y$，传入 `weight` 时令 $g=grad_y\times weight[:,None]$：

$$
grad_a = g \times (b_c+bias) \times s \times (1+\alpha a_c(1-s)) \times mask_a
$$

$$
grad_b = g \times a_c \times s \times mask_b
$$

最终 `grad_x` 为 `grad_a` 和 `grad_b` 沿最后一维的拼接结果。仅当同时传入 `weight` 和 `y_origin` 时计算：

$$
grad\_weight[t] = \sum_{h=0}^{H-1} grad_y[t,h] \times y_origin[t,h]
$$

量化以 32 个元素为一个 MX block，scale 采用二次幂取整：

$$
scale = \mathrm{ceil\_power\_of\_two}\left(\frac{amax}{fp8\_max}\right),
\qquad output = \mathrm{cast}_{dst\_type}\left(\frac{grad_x}{scale}\right)
$$

`dst_type=35` 使用 FLOAT8_E5M2，`dst_type=36` 使用 FLOAT8_E4M3FN。`scale1` 是沿最后一维的量化缩放因子，`scale2` 是沿倒数第二维的量化缩放因子。group 场景的 `scale2` 采用带 group 边界偏移的布局，额外行用于承载分组边界，不是普通无效 padding。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|:---|:---|:---|:---|:---|
|grad_y|输入|SwiGLU 输出的反向梯度，shape 为 `[T, H]`。|FLOAT16、BFLOAT16|ND|
|x|输入|SwiGLU 前向输入，shape 为 `[T, 2H]`，最后一维必须为 64 的整数倍。|FLOAT16、BFLOAT16|ND|
|weight|可选输入|每个 token 的权重，shape 为 `[T]`；仅 group 场景支持。|FLOAT16、BFLOAT16、FLOAT|ND|
|y_origin|可选输入|SwiGLU 前向输出的原始值，shape 为 `[T, H]`；与 `weight` 成对传入时用于计算 `grad_weight`。|FLOAT16、BFLOAT16|ND|
|group_index|可选输入|group 的累计终点，shape 为 `[G]`，采用 cumsum 模式。|INT64|ND|
|clamp_limit|可选属性|Clamp 阈值，默认值为 `-1.0`；`-1.0` 表示不启用 Clamp，启用时必须大于 0。|FLOAT|—|
|alpha|可选属性|SwiGLU 反向计算中的 alpha 系数，默认值为 `1.0`。|FLOAT|—|
|bias|可选属性|SwiGLU 反向计算中的 bias 系数，默认值为 `0.0`。|FLOAT|—|
|quant_mode|可选属性|量化模式，当前仅支持 `1`，表示动态 MX 量化。|INT64|—|
|dst_type|可选属性|FP8 输出类型，`35` 表示 FLOAT8_E5M2，`36` 表示 FLOAT8_E4M3FN，默认值为 `36`。|INT64|—|
|y1|输出|沿最后一维量化后的结果，shape 与 `x` 相同。|FLOAT8_E4M3FN、FLOAT8_E5M2|ND|
|scale1|输出|沿最后一维的 E8M0 缩放因子，shape 为 `[T, ceil(2H/64), 2]`。|FLOAT8_E8M0|ND|
|y2|输出|沿倒数第二维量化后的结果，shape 与 `x` 相同。|FLOAT8_E4M3FN、FLOAT8_E5M2|ND|
|scale2|输出|沿倒数第二维的 E8M0 缩放因子；非 group 为 `[ceil(T/64), 2H, 2]`，group 为 `[floor(T/64)+G, 2H, 2]`。|FLOAT8_E8M0|ND|
|grad_weight|可选输出|`weight` 的梯度，仅在同时传入 `weight` 和 `y_origin` 时输出，shape 和数据类型与 `weight` 相同。|FLOAT16、BFLOAT16、FLOAT|ND|

## 约束说明

- 仅支持 Ascend 950；输入 `x` 和 `grad_y` 必须为二维 Tensor，且 shape 分别为 `[T, 2H]` 和 `[T, H]`。
- `x` 的最后一维必须为 64 的整数倍，`grad_y` 的最后一维必须等于 `x` 最后一维的一半。
- 非 group 场景不传入 `group_index`、`weight` 和 `y_origin`；group 场景必须传入 `group_index`。
- `group_index` 采用 cumsum 模式，必须满足严格递增且最后一个值等于 `T`。
- `weight` 仅支持 group 场景；传入 `weight` 时必须同时传入 `y_origin`，二者的 shape 和数据类型需满足参数说明。
- `quant_mode` 仅支持 `1`；`dst_type` 仅支持 `35` 和 `36`。
- `clamp_limit` 仅支持 `-1.0` 或大于 0 的值，`alpha` 必须大于 0。
- 输入 Tensor 支持非连续格式，算子原型通过 `AutoContiguous` 处理；输入输出数据格式为 ND。
- 不支持 `interleaved`、`dim` 等未定义参数，也不支持 Batch 一致性配置。

## 调用说明

|调用方式|调用样例|说明|
|:---|:---|:---|
|aclnn 调用|[test_aclnn_swiglu_backward_group_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_backward_group_quant_with_dual_axis.cpp)|通过 [aclnnSwigluBackwardGroupQuantWithDualAxis](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md) 接口调用算子。|
|图模式调用|—|通过[算子 IR](./op_graph/swiglu_backward_group_quant_with_dual_axis_proto.h)定义算子图节点。|
|PyTorch API|—|通过 [swiglu_backward_group_quant_with_dual_axis](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md) 接口调用算子。|

## 参考资源

- [aclnn API 文档](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md)
- [PyTorch API 文档](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md)
