# NllLossGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：负对数似然损失反向。
- 计算公式：

$$
x\_grad[n][i] =
\begin{cases}
    \displaystyle -y\_grad[n] \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{none} \\
    \displaystyle -\frac{y\_grad}{\text{total\_weight}} \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{mean} \\
    \displaystyle -y\_grad \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{sum} \\
    0, & \text{if } i \neq \text{target}
\end{cases}
$$

## 参数说明
<table style="undefined;table-layout: fixed; width: 1349px"><colgroup>
      <col style="width: 158px">
      <col style="width: 120px">
      <col style="width: 253px">
      <col style="width: 283px">
      <col style="width: 218px">
      <col style="width: 110px">
      <col style="width: 102px">
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
        <td>x</td>
        <td>输入</td>
        <td>输入概率tensor。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y_grad</td>
        <td>输入</td>
        <td>正向结果的梯度，对应公式中的y_grad。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td>表示真实标签，公式中的target。</td>
        <td>INT64、UINT8、INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>表示每个类别的缩放权重，公式中的weight。</td>
        <td>数据类型和self保持一致。</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>输入</td>
        <td>指定要应用到输出的缩减，公式中的reduction。</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>total_weight</td>
        <td>输入</td>
        <td>公式中的total_weight。</td>
        <td>数据类型与weight相同。</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>输入</td>
         <td>指定一个被忽略且不影响输入梯度的目标值。
        </td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x_grad</td>
        <td>输出</td>
        <td>公式中的x_grad[n][i]。</td>
        <td>数据类型和self一致。</td>
        <td>ND</td>
      </tr>
      </tbody>
      </table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_nll_loss_grad](./examples/test_aclnn_nll_loss_grad.cpp) | 通过[aclnnNLLLossBackward](./docs/aclnnNLLLossBackward.md)接口方式调用NLLLossGrad算子。 |
| aclnn调用 | [test_aclnn_nll_loss_grad_2d](./examples/test_aclnn_nll_loss_grad_2d.cpp) | 通过[aclnnNLLLoss2dBackward](./docs/aclnnNLLLoss2dBackward.md)接口方式调用NLLLossGrad算子。 |