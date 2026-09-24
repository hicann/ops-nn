# SwigluBackwardGroupQuantWithDualAxis

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

### 接口功能

SwigluBackwardGroupQuantWithDualAxis融合带可选Clamp和weight的SwiGLU反向计算，以及沿最后一维和倒数第二维的动态MX量化。当前仅支持`quantMode=1`，输出FP8数据和E8M0缩放因子。

- 非 group 场景：不传入 `group_index`、`weight` 和 `y_origin`，返回 `y1`、`scale1`、`y2` 和 `scale2`。
- group 场景：传入 cumsum 模式的 `group_index`，可选成对传入 `weight` 和 `y_origin`；传入二者时额外返回 `grad_weight`。
- 算子仅支持二维输入：`x`的shape为`[T, 2H]`，`gradY`的shape为`[T, H]`。

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

其中，未启用Clamp时`mask_a`和`mask_b`均为1；启用时分别表示对应输入未被截断。最终`grad_x`为`grad_a`和`grad_b`沿最后一维的拼接结果。仅当同时传入`weight`和`y_origin`时计算：

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

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| gradY | 输入 | SwiGLU输出的反向梯度，shape为`[T, H]`。 | FLOAT16、BFLOAT16 | ND |
| x | 输入 | SwiGLU前向输入，shape为`[T, 2H]`。 | FLOAT16、BFLOAT16 | ND |
| weightOptional | 可选输入 | 每个token的权重，shape为`[T]`；仅group场景支持。 | FLOAT16、BFLOAT16、FLOAT | ND |
| yOriginOptional | 可选输入 | SwiGLU前向输出的原始值，shape为`[T, H]`；与weight成对传入。 | FLOAT16、BFLOAT16 | ND |
| groupIndexOptional | 可选输入 | group累计终点，shape为`[G]`，采用cumsum模式。 | INT64 | ND |
| clampLimit | 属性 | Clamp阈值；`-1.0`表示不启用，启用时大于0。 | DOUBLE | - |
| alpha | 属性 | SwiGLU反向计算中的alpha系数，必须大于0。 | DOUBLE | - |
| bias | 属性 | SwiGLU反向计算中的bias系数。 | DOUBLE | - |
| quantMode | 属性 | 量化模式，仅支持`1`。 | INT64 | - |
| dstType | 属性 | FP8类型：`35`为FLOAT8_E5M2，`36`为FLOAT8_E4M3FN。 | INT64 | - |
| y1Out | 输出 | -1轴量化结果，shape与x相同。 | FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| scale1Out | 输出 | -1轴E8M0缩放因子，shape为`[T, ceil(2H/64), 2]`。 | FLOAT8_E8M0 | ND |
| y2Out | 输出 | -2轴量化结果，shape与x相同。 | FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| scale2Out | 输出 | -2轴E8M0缩放因子；非group为`[ceil(T/64), 2H, 2]`，group为`[floor(T/64)+G, 2H, 2]`。 | FLOAT8_E8M0 | ND |
| gradWeightOutOptional | 可选输出 | weight梯度，仅与weight、yOrigin成对使用，shape和类型与weight相同。 | FLOAT16、BFLOAT16、FLOAT | ND |

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

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_swiglu_backward_group_quant_with_dual_axis](./examples/arch35/test_aclnn_swiglu_backward_group_quant_with_dual_axis.cpp) | 通过[aclnnSwigluBackwardGroupQuantWithDualAxis](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md)接口调用算子。 |
| 图模式调用 | - | 通过[算子IR](./op_graph/swiglu_backward_group_quant_with_dual_axis_proto.h)定义算子图节点。 |
| PyTorch API | - | 通过[swiglu_backward_group_quant_with_dual_axis](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md)接口调用算子。 |

## 参考资源

- [aclnn API 文档](./docs/aclnnSwigluBackwardGroupQuantWithDualAxis.md)
- [PyTorch API 文档](./docs/torchapi_swiglu_backward_group_quant_with_dual_axis.md)
