# ApplyAdamV2

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:-------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：实现Adam（Adaptive Moment Estimation，自适应矩估计）优化器的参数更新，支持`adam`和`mbart_adam`两种模式，可选梯度裁剪，用于深度学习训练中反向传播后的参数更新阶段。
- 计算公式：

  **标准Adam模式**（`adam_mode = "adam"`）：

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

  **mBART Adam模式**（`adam_mode = "mbart_adam"`）：

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
  \text{var}_t = \text{var}_{t-1} - \left(\text{step\_size} \cdot u + \text{weight\_decay} \cdot (\eta \cdot \text{var}_{t-1})\right)
  $$

  其中$\mathrm{var}$为模型参数，$m$为一阶矩估计，$v$为二阶矩估计，$\beta_1$、$\beta_2$为衰减率，$\eta$为学习率，$\epsilon$为数值稳定性常数。$\mathrm{clip\_coeff}$为梯度缩放系数，取值规则参见[约束说明](#约束说明)。$\mathrm{weight\_decay}$为权重衰减系数，$\mathrm{step\_size}$为更新步长。

## 参数说明

Tensor表示张量，shape表示形状，dtype表示数据类型。ND表示多维Tensor的数据格式，具体参见[数据格式](../../docs/zh/context/data_format.md)。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:-------|:--------------|:-----|:--------|:--------|
| var | 输入 | 待更新的模型参数，对应公式中$\mathrm{var}_{t-1}$。 | FLOAT、FLOAT16 | ND |
| m | 输入 | 一阶矩估计，对应公式中$m_{t-1}$。shape与var一致。 | FLOAT、FLOAT16 | ND |
| v | 输入 | 二阶矩估计，对应公式中$v_{t-1}$。shape与var一致。 | FLOAT、FLOAT16 | ND |
| lr | 输入 | 学习率，对应公式中$\eta$，标量。 | FLOAT、FLOAT16 | ND |
| beta1 | 输入 | 一阶矩衰减率，对应公式中$\beta_1$，标量，典型值0.9。 | FLOAT、FLOAT16 | ND |
| beta2 | 输入 | 二阶矩衰减率，对应公式中$\beta_2$，标量，典型值0.999。 | FLOAT、FLOAT16 | ND |
| epsilon | 输入 | 数值稳定性常数，对应公式中$\epsilon$，标量，典型值`1e-8`。 | FLOAT、FLOAT16 | ND |
| grad | 输入 | 梯度张量。shape与var一致。 | FLOAT、FLOAT16 | ND |
| max_grad_norm | 输入（可选） | 梯度裁剪控制标量，值大于0时启用。仅`adam`模式使用。 | FLOAT、FLOAT16 | ND |
| global_grad_norm | 输入 | 全局梯度范数，标量。仅`adam`模式使用。 | FLOAT、FLOAT16 | ND |
| weight_decay | 输入 | 权重衰减系数，标量，典型值0.01。 | FLOAT、FLOAT16 | ND |
| step_size | 输入（可选） | 更新步长，标量。仅`mbart_adam`模式使用。 | FLOAT、FLOAT16 | ND |
| adam_mode | 属性 | 计算模式。`"adam"`表示标准Adam模式；`"mbart_adam"`表示mBART Adam模式。默认值为`"adam"`。 | STRING | - |
| var | 输出 | 更新后的模型参数，对应公式中$\mathrm{var}_t$。shape和dtype与输入var一致。 | FLOAT、FLOAT16 | ND |
| m | 输出 | 更新后的一阶矩估计，对应公式中$m_t$。shape和dtype与输入m一致。 | FLOAT、FLOAT16 | ND |
| v | 输出 | 更新后的二阶矩估计，对应公式中$v_t$。shape和dtype与输入v一致。 | FLOAT、FLOAT16 | ND |

## 约束说明

- var、m、v、grad四个Tensor的shape和数据类型必须完全一致。
- lr、beta1、beta2、epsilon、max_grad_norm、global_grad_norm、weight_decay、step_size为标量Tensor，shape支持`[]`或`[1]`，数据类型必须与var的数据类型一致。
- FLOAT16（16位浮点数）输入时，内部使用FLOAT32（32位浮点数）计算，输出再转回FLOAT16。
- `adam`模式下，仅当提供max_grad_norm且其值大于0时启用梯度裁剪。global_grad_norm为非数值（NaN）、无穷大（Inf）或小于`1e-10`时按1处理；处理后若大于1，则`clip_coeff = 1 / global_grad_norm`，否则`clip_coeff = 1`。未启用裁剪时`clip_coeff = 1`。max_grad_norm用作裁剪开关，其数值不作为缩放分子。
- `mbart_adam`模式下不进行梯度裁剪，若未提供step_size，则使用lr作为step_size。
- 算子默认确定性实现，相同输入产生相同输出。
- 合法元数据的空Tensor（任意维度长度为0）返回与输入shape、dtype一致的三个空输出，不进行数值计算。
- Tensor的维度数支持0至8。
- 动态shape支持情况：Kernel模式支持`-1`（未知维度）和`-2`（未知维度数）；图引擎（Graph Engine，GE）图模式不支持`-1`或`-2`，因为canndev内置形状推导无法将未知维度正确传播到原地更新输出。该说明适用于图模式，aclnn接口的shape约束参见[aclnnApplyAdamV2](docs/aclnnApplyAdamV2.md#约束说明)。

### 数值范围

本算子支持符合Adam参数更新语义的输入，调用方应保证以下数值范围：

| 参数 | 支持范围 |
|:-----|:---------|
| var、m、grad | 有限实数，允许正值、负值和0，不额外限制梯度或一阶矩的幅值。 |
| v | 有限且非负，表示梯度平方的二阶矩估计。 |
| beta1、beta2 | 配置值满足`0 <= beta < 1`。FLOAT16输入允许该配置转换时舍入为`1`的边界情况。 |
| lr | 有限且非负。 |
| epsilon | 有限且非负，通常使用正值。 |
| weight_decay | 有限且非负。 |
| step_size | `mbart_adam`模式下有限且非负；`adam`模式下不参与计算，不要求其数值满足此范围。 |

## 调用说明

本仓提供的`aclnnApplyAdamV2`接口支持Ascend 950PR&950DT系列产品。此处描述的是该接口的产品支持范围，接口原型、原地更新语义及参数约束参见[aclnnApplyAdamV2](docs/aclnnApplyAdamV2.md)。

| 调用方式 | 调用样例 | 说明 |
|:--------|:--------|:-----|
| aclnn调用 | [test_aclnn_apply_adam_v2](examples/arch35/test_aclnn_apply_adam_v2.cpp) | 先调用`aclnnApplyAdamV2GetWorkspaceSize`，再调用`aclnnApplyAdamV2`，原地更新var、m、v。 |
| 图模式调用 | [test_geir_apply_adam_v2](examples/test_geir_apply_adam_v2.cpp) | 通过[算子IR](op_graph/apply_adam_v2_proto.h)构图方式调用ApplyAdamV2算子。 |
