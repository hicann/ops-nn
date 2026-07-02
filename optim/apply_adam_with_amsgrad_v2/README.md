# ApplyAdamWithAmsgradV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √     |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √     |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：Adam AMSGrad变体单步原地更新优化器。维护二阶矩历史最大值`vhat`，用`max(v, vhat)`替代`v`抑制学习率振荡。`var`/`m`/`v`/`vhat`原地更新到`var_out`/`m_out`/`v_out`/`vhat_out`。

- 计算公式：

  $$
  lr_{t}=lr\cdot\sqrt{1-\beta_{2}^{t}}\,/\,\left(1-\beta_{1}^{t}\right)
  $$
  $$
  m_{t}=m_{t-1}+\left(1-\beta_{1}\right)\left(g_{t}-m_{t-1}\right)
  $$
  $$
  v_{t}=v_{t-1}+\left(1-\beta_{2}\right)\left(g_{t}^{2}-v_{t-1}\right)
  $$
  $$
  \hat{v}_{t}=\max\left(\hat{v}_{t-1}, v_{t}\right)
  $$
  $$
  \theta_{t}=\theta_{t-1}-lr_{t}\cdot\frac{m_{t}}{\sqrt{\hat{v}_{t}}+\epsilon}
  $$

  其中 `beta1_power`、`beta2_power`为$\beta_{1}^{t}$、$\beta_{2}^{t}$标量；epsilon加法顺序锁定为$\sqrt{\hat{v}_{t}}+\epsilon$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
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
    <tr><td>var</td><td>计算输入/计算输出</td><td>待更新权重，原地更新到var_out</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>m</td><td>计算输入/计算输出</td><td>一阶矩估计，原地更新到m_out</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>v</td><td>计算输入/计算输出</td><td>二阶矩估计，原地更新到v_out</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>vhat</td><td>计算输入/计算输出</td><td>二阶矩历史最大值，原地更新到vhat_out</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>beta1_power</td><td>计算输入</td><td>β1的t次幂，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>beta2_power</td><td>计算输入</td><td>β2的t次幂，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>lr</td><td>计算输入</td><td>学习率，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>beta1</td><td>计算输入</td><td>一阶矩衰减系数，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>beta2</td><td>计算输入</td><td>二阶矩衰减系数，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>epsilon</td><td>计算输入</td><td>防止除零，标量 (1,)</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>grad</td><td>计算输入</td><td>梯度，与var同shape，FLOAT32</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>var_out</td><td>计算输出</td><td>更新后权重，inplace var</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>m_out</td><td>计算输出</td><td>更新后一阶矩，inplace m</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>v_out</td><td>计算输出</td><td>更新后二阶矩，inplace v</td><td>FLOAT32</td><td>ND</td></tr>
    <tr><td>vhat_out</td><td>计算输出</td><td>更新后历史最大值，inplace vhat</td><td>FLOAT32</td><td>ND</td></tr>
  </tbody></table>

## 约束说明

- var/m/v/vhat/grad 数据类型须一致，仅支持FLOAT32；标量输入固定FLOAT32。
- beta1_power、beta2_power、lr、beta1、beta2、epsilon 的shape须为 (1,)。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_apply_adam_with_amsgrad_v2](./examples/arch35/test_geir_apply_adam_with_amsgrad_v2.cpp) | 通过图模式方式调用ApplyAdamWithAmsgradV2 算子。 |
