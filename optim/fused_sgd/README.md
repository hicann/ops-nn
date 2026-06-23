# FusedSgd

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：实现fusedSgd算子。将传统 SGD 更新过程中原本分散的多个细粒度操作（如梯度缩放、权重衰减、动量更新、参数赋值等）融合为单个NPU Kernel执行。
- **计算公式**：

  $$
  \begin{aligned} 
  &\tilde{g}_t = \begin{cases} 
  g_t / s & s \neq \text{None} \\ 
  g_t & \text{otherwise} 
  \end{cases} \\ 

  &\hat{g}_t = \begin{cases} 
  -\tilde{g}_t & \text{maximize} \\ 
  \tilde{g}_t & \text{otherwise} 
  \end{cases} \\ 

  &\bar{g}_t = \hat{g}_t + weightDecay \cdot \theta_t \\ 

  &v_{t+1} = \begin{cases} 
  \bar{g}_t & \text{first step} \\ 
  \mu v_t + (1-dampening)\bar{g}_t & \text{otherwise} 
  \end{cases} \\ 

  &g_t^{\text{final}} = \begin{cases} 
  \bar{g}_t + \mu v_{t+1} & \text{nesterov} \\ 
  v_{t+1} & v_t \neq \text{None} \\ 
  \bar{g}_t & \text{otherwise}
  \end{cases} \\ 

  &\theta_{t+1} = \theta_t - lr \cdot g_t^{\text{final}} \\
  &g_{t+1} = \tilde{g}_t

  \end{aligned}
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
<col style="width: 155px">
<col style="width: 162px">
<col style="width: 380px">
<col style="width: 276px">
<col style="width: 107px">
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
    <td>params</td>
    <td>输入/输出</td>
    <td>待更新参数，对应公式中的θ。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grads</td>
    <td>输入/输出</td>
    <td>待更新参数对应的梯度，对应公式中的g。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待更新参数对应的动量，对应公式中的v。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradScale</td>
    <td>输入</td>
    <td>梯度缩放大小，对应公式中的s。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>待更新参数对应的动量，对应公式中的v。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightDecay</td>
    <td>属性</td>
    <td>权重衰减值，对应公式中的weightDecay，默认为0。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>属性</td>
    <td>动量值，对应公式中的μ，默认为0。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>属性</td>
    <td>学习率，对应公式中的lr，默认为1e-3。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dampening</td>
    <td>属性</td>
    <td>动量阻尼系数，对应公式中的dampening，默认为0。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>nesterov</td>
    <td>属性</td>
    <td>是否启用Nesterov动量，对应公式中的nesterov，默认为False。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maximize</td>
    <td>属性</td>
    <td>是否为最大化目标函数，对应公式中的maximize，默认为False。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>isFirstStep</td>
    <td>属性</td>
    <td>是否第一步更新，对应公式中的FirstStep，默认为True。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- params、grads、x、gradScale的数据类型在支持的范围之内。
- params、grads与x及其中各个tensor具有相同的数据类型。
- params、grads与x中tensor的shape维度小于等于8，gradScale的shape为[1]。
- lr、momentum、weightDecay、dampening的值大于等于0。
- params、grads与x（x不为空时）中相同索引tensor的shape相同。
- params、grads与x（x不为空时）中不能有空指针。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_fused_sgd](./examples/test_aclnn_fused_sgd.cpp) | 通过[aclnnFusedSgd](docs/aclnnFusedSgd.md)接口方式调用FusedSgd算子。 |
