# ApplyAdamV2

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

- 算子功能：实现 Adam 优化器的参数更新，支持标准 Adam 和 mBART Adam 两种模式，可选梯度裁剪，用于深度学习训练中反向传播后的参数更新阶段。
- 计算公式：

  **标准 Adam 模式**（adam_mode = "adam"）：

  $$
  g_t = \text{grad} \cdot \text{clip\_coeff}
  $$

  $$
  m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
  $$

  $$
  v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
  $$

  $$
  u = \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

  $$
  \text{var}_t = \text{var}_{t-1} - \eta \cdot (u + \text{weight\_decay} \cdot \text{var}_{t-1})
  $$

  **mBART Adam 模式**（adam_mode = "mbart_adam"）：

  $$
  m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \text{grad}
  $$

  $$
  v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot \text{grad}^2
  $$

  $$
  u = \frac{m_t}{\sqrt{v_t} + \epsilon}
  $$

  $$
  \text{var}_t = \text{var}_{t-1} - \text{step\_size} \cdot u - \eta \cdot \text{weight\_decay} \cdot \text{var}_{t-1}
  $$

  其中 $var$ 为模型参数，$m$ 为一阶矩估计，$v$ 为二阶矩估计，$\beta_1$、$\beta_2$ 为衰减率，$\eta$ 为学习率，$\epsilon$ 为数值稳定性常数。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| var | 输入 | 待更新的模型参数，对应公式中 var<sub>t-1</sub>。 | FLOAT、FLOAT16 | ND |
| m | 输入 | 一阶矩估计，对应公式中 m<sub>t-1</sub>。shape 与 var 一致。 | FLOAT、FLOAT16 | ND |
| v | 输入 | 二阶矩估计，对应公式中 v<sub>t-1</sub>。shape 与 var 一致。 | FLOAT、FLOAT16 | ND |
| lr | 输入 | 学习率，对应公式中 η，标量。 | FLOAT、FLOAT16 | ND |
| beta1 | 输入 | 一阶矩衰减率，对应公式中 β<sub>1</sub>，标量，典型值 0.9。 | FLOAT、FLOAT16 | ND |
| beta2 | 输入 | 二阶矩衰减率，对应公式中 β<sub>2</sub>，标量，典型值 0.999。 | FLOAT、FLOAT16 | ND |
| epsilon | 输入 | 数值稳定性常数，对应公式中 ε，标量，典型值 1e-8。 | FLOAT、FLOAT16 | ND |
| grad | 输入 | 梯度张量。shape 与 var 一致。 | FLOAT、FLOAT16 | ND |
| max_grad_norm | 输入（可选） | 梯度裁剪阈值，标量。仅 adam 模式使用。 | FLOAT、FLOAT16 | ND |
| global_grad_norm | 输入 | 全局梯度范数，标量。仅 adam 模式使用。 | FLOAT、FLOAT16 | ND |
| weight_decay | 输入 | 权重衰减系数，标量，典型值 0.01。 | FLOAT、FLOAT16 | ND |
| step_size | 输入（可选） | 步长缩放因子，标量。仅 mbart_adam 模式使用。 | FLOAT、FLOAT16 | ND |
| adam_mode | 属性 | <ul><li>计算模式。</li><li>"adam"：标准 Adam 模式。</li><li>"mbart_adam"：mBART Adam 模式。</li><li>默认值为 "adam"。</li></ul> | STRING | - |
| var | 输出 | 更新后的模型参数，对应公式中 var<sub>t</sub>。shape 和 dtype 与输入 var 一致。 | FLOAT、FLOAT16 | ND |
| m | 输出 | 更新后的一阶矩估计，对应公式中 m<sub>t</sub>。shape 和 dtype 与输入 m 一致。 | FLOAT、FLOAT16 | ND |
| v | 输出 | 更新后的二阶矩估计，对应公式中 v<sub>t</sub>。shape 和 dtype 与输入 v 一致。 | FLOAT、FLOAT16 | ND |

## 约束说明

- var、m、v、grad 四个 Tensor 的 shape 和数据类型必须完全一致。
- lr、beta1、beta2、epsilon、max_grad_norm、global_grad_norm、weight_decay、step_size 为标量（shape [1]），数据类型必须与 Tensor 的数据类型一致。
- FP16 输入时，内部计算提升到 FP32 进行，输出再转回 FP16，以保证长期训练中矩估计的精度。
- adam 模式下，若提供 max_grad_norm 和 global_grad_norm，则进行梯度裁剪：clip_coeff = min(1.0, max_grad_norm / global_grad_norm)。
- mbart_adam 模式下，若未提供 step_size，则使用 lr 作为 step_size。
- 算子默认确定性实现，相同输入产生相同输出。
- 不支持空 Tensor（0 元素）。
- Tensor rank 范围 0~8。
- 动态 shape 支持情况：Kernel 模式支持 -1/-2（未知维度）；GEIR 模式不支持 -1/-2，因为 canndev 内置 infershape 无法将未知维度正确传播到 inplace 输出。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| 图模式调用 | [test_geir_apply_adam_v2](examples/test_geir_apply_adam_v2.cpp) | 通过[算子IR](op_graph/apply_adam_v2_proto.h)构图方式调用ApplyAdamV2算子。 |
