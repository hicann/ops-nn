# FusedAdam

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：

实现Adam优化器功能，支持多组参数列表（TensorList）一次调用完成Adam优化器功能。

- **计算公式**：

$$
\begin{aligned}
&t=t+1 \\

&\tilde{g}_t = \begin{cases}
g_t / s & s \neq \text{None} \\
g_t & \text{otherwise}
\end{cases} \\

&g_{t+1} = \tilde{g}_t \\

&\hat{g}_t = \begin{cases}
-\tilde{g}_t & \text{maximize} \\
\tilde{g}_t & \text{otherwise}
\end{cases} \\

&\bar{g}_t = \hat{g}_t + \lambda \cdot \theta_t \\

&m_t=\beta_1 m_{t-1} + (1-\beta_1) \bar{g}_t\\

&v_t=\beta_2 v_{t-1} + (1-\beta_2) \bar{g}_t^2\\

&max\_v_t= \begin{cases} \max(v_t,max\_v_{t-1}) & \text{amsgrad} \\
max\_v_{t-1} & \text{otherwise}
\end{cases} \\

&\hat{m}_t=\frac{m_t}{1-\beta_1^t}\\

&\hat{v}_t= \begin{cases} \frac{max\_v_t}{1-\beta_2^t} & \text{amsgrad} \\
\frac{v_t}{1-\beta_2^t} & \text{otherwise}
\end{cases} \\

&\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t \\
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
    <td>paramsRef（aclTensorList*）</td>
    <td>输入/输出</td>
    <td><ul><li>不支持空Tensor。</li><li>待计算的权重列表，公式中的θ。</li></ul></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradsRef（aclTensorList*）</td>
    <td>输入/输出</td>
    <td><ul><li>不支持空Tensor。</li><li>梯度数据列表，公式中的g<sub>t</sub>，仅在gradScale输入非空的时候会更新梯度。</li></ul></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expAvgsRef（aclTensorList*）</td>
    <td>输入/输出</td>
    <td><ul><li>不支持空Tensor。</li><li>一阶动量列表，公式中的m。</li></ul></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>expAvgSqsRef（aclTensorList*）</td>
    <td>输入/输出</td>
    <td><ul><li>不支持空Tensor。</li><li>二阶动量列表，公式中的v，不能为负数。</li></ul></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>maxExpAvgSqsRef（aclTensorList*）</td>
    <td>输入/输出</td>
    <td><ul><li>不支持空Tensor。</li><li>保存最大二阶矩列表，与更新后的expAvgSqsRef比较后取最大值输出。</li><li>此参数在amsgrad参数为true时必选，在amsgrad参数为false时可选。</li></ul></td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>stateSteps（aclTensorList*）</td>
    <td>输入</td>
    <td><ul><li>不支持空Tensor。</li><li>迭代次数列表，公式中的t，需要大于0。</li></ul></td>
    <td>INT64、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradScaleOptional（aclTensor*）</td>
    <td>输入</td>
    <td>可选输入，梯度缩放因数s。当gradScaleOptional非空时，会据此更新并输出梯度（覆盖原有梯度）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>foundInfOptional（aclTensor*）</td>
    <td>输入</td>
    <td>可选输入，标识是否出现Inf/NaN。当foundInfOptional等于1时停止更新。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr（double）</td>
    <td>属性</td>
    <td>学习率，公式中的η。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta1（double）</td>
    <td>属性</td>
    <td>β<sub>1</sub>参数。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>beta2（double）</td>
    <td>属性</td>
    <td>β<sub>2</sub>参数。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>weightDecay（double）</td>
    <td>属性</td>
    <td>权重衰减系数，公式中的λ。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>eps（double）</td>
    <td>属性</td>
    <td>防止除数为0。</td>
    <td>DOUBLE</td>
    <td>-</td>
  </tr>
  <tr>
    <td>amsgrad（bool）</td>
    <td>属性</td>
    <td>是否使用算法的AMSGrad变量。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maximize（bool）</td>
    <td>属性</td>
    <td>是否最大化参数。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- 输入paramsRef、gradsRef、expAvgsRef、expAvgSqsRef这些tensorList中每个tensor不得为空，数据类型必须一致，且数据类型仅支持FLOAT16、BFLOAT16、FLOAT32。

- 输入tensorList中paramsRef、gradsRef、expAvgsRef、expAvgSqsRef中，tensor个数必须保持一致，且下标相同的tensor的shape必须保持一致。

- stateSteps类型为tensorList，支持INT64、FLOAT32，其tensor个数必须和paramsRef、gradsRef、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef一致。每个tensor元素个数至少为1，如果元素个数大于1则取第0个元素的值作为stateSteps的值。

- amsgrad为false时，maxExpAvgSqsRef可为空；amsgrad为true时，maxExpAvgSqsRef必选tensor数量，每个tensor的shape和dtype必须与paramsRef、gradsRef、expAvgsRef、expAvgSqsRef一致。

- 确定性计算：
  - aclnnFusedAdam默认确定性实现。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_fused_adam](./examples/arch35/test_aclnn_fused_adam.cpp) | 通过[aclnnFusedAdam](docs/aclnnFusedAdam.md)接口方式调用FusedAdam算子。 |
