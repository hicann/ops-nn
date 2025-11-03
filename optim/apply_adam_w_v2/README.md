# ApplyAdamWV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：
  
  实现adamW优化器功能。

- 计算公式：
  $$
  if(maximize) : g_{t} = - g_{t}
  $$
  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}
  $$
  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$
  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}
  $$
  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
  $$
  $$
  if(amsgard) : maxGradNorm = max(maxGradNorm,\hat{v}_{t})
  $$
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 370px">
  <col style="width: 200px">
  <col style="width: 70px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>varRef</td>
      <td>输入/输出</td>
      <td>待计算的权重输入同时也是输出，公式中的输入/输出θ 
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mRef</td>
      <td>输入/输出</td>
      <td>待更新参数对应的一阶动量，公式中的输入/输出m。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>vRef</td>
      <td>输入/输出</td>
      <td>待更新参数对应的二阶动量，公式中的输入/输出v。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxGradNormOptionalRef</td>
      <td>输入/输出</td>
      <td>保存v参数的最大值，公式中的maxGradNorm。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度数据，公式中的输入g。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>step</td>
      <td>输入</td>
      <td>迭代次数，公式中的t。</td>
      <td>INT64、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>属性</td>
      <td><ul><li>学习率。</li><li>取值范围是(0,1)，默认为0.1。计算公式中的η。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1
      <td>属性</td>
      <td><ul><li>beta1参数。</li><li>取值范围是(0,1)，默认为0.1。计算公式中的β1。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>属性</td>
      <td><ul><li>beta2参数。</li><li>取值范围是(0,1)，默认为0.1。计算公式中的β2。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightDecay</td>
      <td>属性</td>
      <td><ul><li>权重衰减系数。</li><li>取值范围是(0,1)，默认为0.1。计算公式中的λ。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td><ul><li>防除0参数。</li><li>默认为1e-8。计算公式中的ϵ。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>amsgrad</td>
      <td>属性</td>
      <td><ul><li>是否使用算法的AMSGrad变量，默认为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maximize</td>
      <td>属性</td>
      <td><ul><li>是否最大化参数，默认为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无  

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_apply_adam_w_v2](./examples/test_aclnn_apply_adam_w_v2.cpp) | 通过[aclnnApplyAdamWV2](./docs/aclnnApplyAdamWV2.md)接口方式调用ApplyAdamWV2算子。    |
