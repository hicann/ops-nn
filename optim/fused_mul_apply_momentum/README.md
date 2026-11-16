# FusedMulApplyMomentum

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：实现 SGD with Momentum 优化器的参数更新，融合梯度乘法（`grad = x1 * x2`），用于深度学习训练中反向传播后的参数更新阶段，适用于混合精度训练场景下的梯度缩放（Loss Scaling）。
- 计算公式：

  **标准模式**（use_nesterov = false）：

  $$
  \text{accum}_{t} = \text{accum}_{t-1} \cdot m + x_1 \cdot x_2
  $$

  $$
  \text{var}_{t} = \text{var}_{t-1} - \text{accum}_{t} \cdot \eta
  $$

  **Nesterov 模式**（use_nesterov = true）：

  $$
  \text{accum}_{t} = \text{accum}_{t-1} \cdot m + x_1 \cdot x_2
  $$

  $$
  \text{var}_{t} = \text{var}_{t-1} - x_1 \cdot x_2 \cdot \eta - \text{accum}_{t} \cdot m \cdot \eta
  $$

  其中 $var$ 为模型参数，$accum$ 为动量累积器，$x_1$ 为梯度张量，$x_2$ 为梯度缩放因子，$m$ 为动量系数，$\eta$ 为学习率。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| var | 输入 | 待更新的模型参数，对应公式中 var<sub>t-1</sub>。 | FLOAT、FLOAT16 | ND |
| accum | 输入 | 动量累积器，对应公式中 accum<sub>t-1</sub>。shape 与 var 相同。 | FLOAT、FLOAT16 | ND |
| lr | 输入 | 学习率，对应公式中 η，单元素Tensor。 | FLOAT、FLOAT16 | ND |
| x1 | 输入 | 梯度张量，对应公式中 x<sub>1</sub>。shape 与 var 相同。 | FLOAT、FLOAT16 | ND |
| momentum | 输入 | 动量系数，对应公式中 m，单元素Tensor，典型值 0.9。 | FLOAT、FLOAT16 | ND |
| x2 | 输入 | 梯度缩放因子，对应公式中 x<sub>2</sub>，单元素Tensor。 | FLOAT、FLOAT16 | ND |
| use_nesterov | 属性 | <ul><li>Nesterov模式开关。</li><li>默认值为false。</li></ul> | BOOL | - |
| use_locking | 属性 | <ul><li>锁控制（TensorFlow兼容保留，不使用）。</li><li>默认值为false。</li></ul> | BOOL | - |
| var | 输出 | 更新后的模型参数，对应公式中 var<sub>t</sub>。shape 和 dtype 与 var 相同。 | FLOAT、FLOAT16 | ND |
| accum | 输出 | 更新后的动量累积器，对应公式中 accum<sub>t</sub>。shape 和 dtype 与 var 相同。 | FLOAT、FLOAT16 | ND |

## 约束说明

- 直接调用Kernel时，输入需为连续Tensor。

- var、accum、x1 三个 Tensor 的 shape 和数据类型必须完全一致。
- lr、momentum、x2 为单元素Tensor（可为0维或其他元素数为1的shape），数据类型必须与 Tensor 的数据类型一致。
- 仅 lr、momentum、x2 三个标量支持广播参与计算；var、accum、x1 之间不支持广播，shape 必须完全一致。
- 支持ND格式。
- use_locking属性为接口兼容保留，不使用。
- 算子默认确定性实现，相同输入产生相同输出。
- 满足上述输入约束的空Tensor返回空结果。
- 算子不限制输入输出的维数。
- 图模式支持动态shape（-1）和未知rank（-2）。

- 不限制浮点输入的取值范围。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:--------|:--------|:-----|
| GE图模式 | [test_geir_fused_mul_apply_momentum](examples/test_geir_fused_mul_apply_momentum.cpp) | 通过[算子IR](op_graph/fused_mul_apply_momentum_proto.h)构图方式调用FusedMulApplyMomentum算子。 |

示例编译与运行方法参见[快速算子调用（GE图模式）](../../docs/zh/invocation/quick_op_invocation.md)。
