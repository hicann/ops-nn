# NllLossGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Negative log likelihood loss backward.
- Formula:

$$
x\_grad[n][i] =
\begin{cases}
    \displaystyle -y\_grad[n] \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{none} \\
    \displaystyle -\frac{y\_grad}{\text{total\_weight}} \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{mean} \\
    \displaystyle -y\_grad \cdot \text{weight}, & \text{if } i = \text{target} \text{ 且 } \text{reduction} = \text{sum} \\
    0, & \text{if } i \neq \text{target}
\end{cases}
$$

## Parameters

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
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>Input probability tensor.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y_grad</td>
        <td>Input</td>
        <td>Gradient of the forward result, corresponding to y_grad in the formulas.</td>
        <td>FLOAT, FLOAT16, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td>Represents the ground-truth label; it is target in the formulas.</td>
        <td>INT64, UINT8, INT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>Input</td>
        <td>Represents the scaling weight of each class; it is weight in the formulas.</td>
        <td>Same data type as self.</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>Input</td>
        <td>Specifies the reduction to apply to the output; it is reduction in the formulas.</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>total_weight</td>
        <td>Input</td>
        <td>It is total_weight in the formulas.</td>
        <td>Same data type as weight.</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>ignore_index</td>
        <td>Input</td>
         <td>Specifies a target value that is ignored and does not affect the input gradient.
        </td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x_grad</td>
        <td>Output</td>
        <td>It is x_grad[n][i] in the formulas.</td>
        <td>Same data type as self.</td>
        <td>ND</td>
      </tr>
      </tbody>
      </table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_nll_loss_grad](./examples/test_aclnn_nll_loss_grad.cpp) | Invokes the NLLLossGrad operator through the [aclnnNLLLossBackward](./docs/aclnnNLLLossBackward_en.md) interface. |
| aclnn invocation | [test_aclnn_nll_loss_grad_2d](./examples/test_aclnn_nll_loss_grad_2d.cpp) | Invokes the NLLLossGrad operator through the [aclnnNLLLoss2dBackward](./docs/aclnnNLLLoss2dBackward_en.md) interface. |
