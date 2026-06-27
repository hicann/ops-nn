# SeluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对输入Tensor计算SELU（Scaled Exponential Linear Unit）激活函数的反向梯度。
- 计算公式：

  $$
  y = \begin{cases} \text{scale} \times \text{gradients}, & \text{outputs} \ge 0 \\ \text{gradients} \times (\text{outputs} + \text{scale} \times \alpha), & \text{outputs} < 0 \end{cases}
  $$

  其中：

  - $\alpha = 1.6732632423543772848170429916717$
  - $\text{scale} = 1.0507009873554804934193349852946$
  - $\text{scale} \times \alpha = 1.7580993408473768599402175208123$

  分段行为：

  - 当 $\text{outputs} \ge 0$ 时：$y = \text{scale} \times \text{gradients}$（线性区梯度）
  - 当 $\text{outputs} < 0$ 时：$y = \text{gradients} \times (\text{outputs} + \text{scale} \times \alpha)$（指数饱和区梯度）

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
  </tr></thead>
<tbody>
  <tr>
    <td>gradients</td>
    <td>输入</td>
    <td>反向传播上游梯度。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
    <td>ND</td>
    <td>1-8</td>
  </tr>
  <tr>
    <td>outputs</td>
    <td>输入</td>
    <td>SELU前向输出，shape与gradients支持numpy广播。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
    <td>ND</td>
    <td>1-8</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>反向梯度结果，shape为gradients与outputs广播后的shape。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT32、INT8、UINT8</td>
    <td>ND</td>
    <td>1-8</td>
  </tr>
</tbody></table>

## 约束说明

- 支持numpy广播：gradients和outputs的shape可以不同，输出y的shape为两者广播后的结果。
- 确定性计算：SeluGrad默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_selu_grad](./examples/arch35/test_aclnn_selu_grad.cpp) | 通过[aclnnSeluBackward](./docs/aclnnSeluBackward.md)接口方式调用SeluGrad算子。 |
