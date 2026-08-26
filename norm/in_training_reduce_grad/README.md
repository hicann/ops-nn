# INTrainingReduceGrad

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | √ |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：实现 Instance Normalization 训练反向梯度计算。在深度学习训练中，前向阶段计算 normalization 的方差和均值，反向阶段根据损失对输入的梯度（dy）计算对原始输入的梯度（pd_x）。本算子接收前向缓存的方差、均值、残差缩放和偏置参数，计算规范化逆变换的完整梯度。
- 计算公式：

  $$
  \text{data\_sqrt} = \sqrt{\text{variance} + \epsilon}
  $$

  $$
  \text{multiplier} = -\frac{\text{res\_gamma}}{N \cdot \text{data\_sqrt}}
  $$

  $$
  \text{addend} = \frac{\text{mean}}{\text{data\_sqrt}} \cdot \frac{\text{res\_gamma}}{N} - \frac{\text{res\_beta}}{N}
  $$

  $$
  \text{coef} = \text{dy} + \text{multiplier} \cdot x + \text{addend}
  $$

  $$
  \text{mul\_scale} = \frac{\text{gamma}}{\text{data\_sqrt}}
  $$

  $$
  \text{pd\_x} = \text{coef} \cdot \text{mul\_scale}
  $$

  其中 $N$ 为空间维度元素数（$H \times W$），$\epsilon$ 为防除零小常数（1e-6），$\text{variance}$ 和 $\text{mean}$ 为前向阶段缓存的统计量，$\text{res\_gamma}$ 和 $\text{res\_beta}$ 为残差缩放和偏置，$\text{gamma}$ 为可学习缩放参数。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| dy | 输入 | 损失对规范化后输出的梯度。 | FLOAT16、FLOAT | NCHW、NHWC |
| x | 输入 | 前向阶段的原始输入。shape 与 dy 一致。 | FLOAT16、FLOAT | NCHW、NHWC |
| variance | 输入 | 前向阶段缓存的方差，通道维度张量。 | FLOAT | NCHW、NHWC |
| mean | 输入 | 前向阶段缓存的均值，通道维度张量。 | FLOAT | NCHW、NHWC |
| res_gamma | 输入 | 残差缩放参数，通道维度张量。 | FLOAT | NCHW、NHWC |
| res_beta | 输入 | 残差偏置参数，通道维度张量。 | FLOAT | NCHW、NHWC |
| gamma | 输入 | 可学习缩放参数，通道维度张量。 | FLOAT | NCHW、NHWC |
| pd_x | 输出 | 对原始输入 x 的梯度。shape 和 dtype 与 dy 一致。 | FLOAT16、FLOAT | NCHW、NHWC |

## 约束说明

- dy 和 x 的 shape 和数据类型必须完全一致。
- variance、mean、res_gamma、res_beta、gamma 的 shape 必须一致，且为通道维度张量：NCHW 格式下 shape 为 [N, C, 1, 1]，NHWC 格式下 shape 为 [N, 1, 1, C]。
- 输入 rank 必须为 4（仅支持 4D NCHW/NHWC）。
- FP16 输入时，内部计算提升到 FP32 进行，输出再转回 FP16（CAST_RINT），以保证梯度计算精度。
- 不支持空 Tensor（H×W=0，即 num=0），此时 Tiling 阶段报错并返回 GRAPH_FAILED。
- 算子默认确定性实现，相同输入产生相同输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| 图模式调用 | [test_geir_in_training_reduce_grad](examples/test_geir_in_training_reduce_grad.cpp) | 通过[算子IR](op_graph/in_training_reduce_grad_proto.h)构图方式调用INTrainingReduceGrad算子。 |
