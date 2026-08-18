# FusedAdamw

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：

  融合AdamW优化器，支持将多组参数列表（params、grads、expAvgs、expAvgSqs、maxExpAvgSqs、stateSteps）的参数更新、一阶/二阶动量更新、梯度缩放等操作融合为单个kernel，提升训练场景下的优化器执行效率。

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
  if(amsgrad) : maxGradNorm = max(maxGradNorm,\hat{v}_{t})
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
      <td>paramsRef</td>
      <td>输入/输出</td>
      <td>待计算的权重列表（TensorList），同时也是输出，公式中的输入/输出θ。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expAvgsRef</td>
      <td>输入/输出</td>
      <td>一阶动量列表（TensorList），公式中的输入/输出m。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>expAvgSqsRef</td>
      <td>输入/输出</td>
      <td>二阶动量列表（TensorList），公式中的输入/输出v。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>maxExpAvgSqsRef</td>
      <td>输入/输出</td>
      <td>最大二阶动量列表（TensorList），公式中的maxGradNorm。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>梯度列表（TensorList），公式中的输入g。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>stateSteps</td>
      <td>输入</td>
      <td>迭代次数列表（TensorList），公式中的t。</td>
      <td>INT64、FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradScaleOptional</td>
      <td>输入</td>
      <td>可选输入，梯度缩放因数。当foundInfOptional=1时，梯度清零并停止参数更新。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>foundInfOptional</td>
      <td>输入</td>
      <td>可选输入，标识是否出现Inf/NaN。等于1时停止更新。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>属性</td>
      <td><ul><li>学习率。</li><li>取值范围是(0,1)，默认为0.001。计算公式中的η。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta1</td>
      <td>属性</td>
      <td><ul><li>beta1参数。</li><li>取值范围是(0,1)，默认为0.9。计算公式中的β1。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta2</td>
      <td>属性</td>
      <td><ul><li>beta2参数。</li><li>取值范围是(0,1)，默认为0.999。计算公式中的β2。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightDecay</td>
      <td>属性</td>
      <td><ul><li>权重衰减系数。</li><li>取值范围是(0,1)，默认为0.0。计算公式中的λ。</li></ul></td>
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

- paramsRef、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef、grads的数据类型必须一致，且支持FLOAT16、BFLOAT16、FLOAT32。
- 同一个TensorList中每个Tensor的shape必须一致；paramsRef、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef、grads对应位置的Tensor的shape也必须一致。
- stateSteps支持INT64、FLOAT32，元素个数为1。
- amsgrad为false时，maxExpAvgSqsRef可为空；amsgrad为true时，maxExpAvgSqsRef必选且shape需与paramsRef一致。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_fused_adamw](./examples/test_aclnn_fused_adamw.cpp) | 通过[aclnnFusedAdamw](./docs/aclnnFusedAdamw.md)接口方式调用FusedAdamw算子。    |
