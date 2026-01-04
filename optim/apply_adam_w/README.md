# ApplyAdamW

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |

## 功能说明

- 算子功能：实现adamW优化器功能。

- 计算公式：

  $$
  g_t=\begin{cases}-g_t
  & \text{ if } maxmize= true\\
  g_t  & \text{ if } maxmize=false
  \end{cases}
  $$
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$
  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$
  $$
  \beta_{1}^{t}=\beta_{1}^{t-1}\times\beta_{1}
  $$
  $$
  \beta_{2}^{t}=\beta_{2}^{t-1}\times\beta_{2}
  $$
  $$
  v_t=\begin{cases}\max(maxGradNorm, v_t)
  & \text{ if } amsgrad = true\\
  v_t  & \text{ if } amsgrad = false
  \end{cases}
  $$
  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$
  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

<table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>计算输入/计算输出</td>
      <td>待计算的权重输入同时也是输出，公式中的theta</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>m</td>
      <td>计算输入/计算输出</td>
      <td>adamw优化器中m参数，公式中的m</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>v</td>
      <td>计算输入/计算输出</td>
      <td>adamw优化器中v参数，公式中的v</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>beta1_power</td>
      <td>计算输入</td>
      <td>beta1^(t-1)参数</td>
      <td>FLOAT16、BFLOAT16、FLOAT32<</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>beta2_power</td>
      <td>计算输入</td>
      <td>beta2^(t-1)参数</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>lr</td>
      <td>计算输入</td>
      <td>学习率，公式中的eta</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>weight_decay</td>
      <td>计算输入</td>
      <td>权重衰减系数</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>beta1</td>
      <td>计算输入</td>
      <td>beta1参数。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>beta2</td>
      <td>计算输入</td>
      <td>beta2参数。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>epsilon</td>
      <td>计算输入</td>
      <td>防止除数为0。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>grad</td>
      <td>计算输入</td>
      <td>梯度数据，公式中的g_t。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    </tr>
      <tr>
      <td>max_grad_norm</td>
      <td>计算输入</td>
      <td>保存v参数的最大值，公式中的v。</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>属性</td>
      <td>是否使用maxGradNormOptional变量</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>属性</td>
      <td>是否对梯度grad取反，应用梯度上升方向优化权重使损失函数最大化</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    </tbody>
  </table>

## 约束说明

- 输入张量的数据类型应保持一致，数据类型支持FLOAT16、BFLOAT16、FLOAT32。
- 输入张量beta1Power、beta2Power、lr、weightDecay、beta1、beta2、eps的shape大小应为1。
- 输入布尔值maximize为true时，maxGradNormOptional参数必选且数据类型和shape应与varRef一致时。

- 确定性计算：
  - aclnnApplyAdamW默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_adam_w](./examples/test_aclnn_apply_adam_w.cpp) | 通过[aclnnApplyAdamW](./docs/aclnnApplyAdamW.md)接口方式调用ApplyAdamW算子。 |