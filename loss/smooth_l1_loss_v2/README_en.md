# SmoothL1LossV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the SmoothL1 loss function.
- Formula:

  For the loss function with a batch of N, when `reduction` is none, this function is defined as:

  $$
  \ell(x,y) = L = \{l_1,\dots,l_N\}^\top
  $$

  where $l_n$ is:

  $$
  l_n = \begin{cases}
  0.5(x_n-y_n)^2/beta, & if |x_n-y_n| < beta \\
  |x_n-y_n| - 0.5*beta, &  otherwise
  \end{cases}
  $$

  If `reduction` is `mean` or `sum`:

  $$
  \ell(x,y)=\begin{cases}
  mean(L), & \text{if reduction} = \text{mean}\\
  sum(L), & \text{if reduction} = \text{sum}
  \end{cases}
  $$

- where:
  - Xn: predictions
  - Yn: labels/truth

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>predict</td>
      <td>Input</td>
      <td>'x' in the formula, indicating the predicted value input used to compute the SmoothL1 loss.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>label</td>
      <td>Input</td>
      <td>'y' in the formula, indicating the true label value input that predict needs to fit.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Indicates the loss value output obtained by computing predict and label through smooth_l1_loss_v2</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>Attribute</td>
      <td>Threshold parameter that controls the boundary between MSE and L1 loss in smooth_l1_loss_v2</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>Reduction parameter that indicates performing mean/sum/no reduction on the element-wise loss values</td>
      <td>STRING</td>
      <td>-</td>
    </tr>

  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_smooth_l1_loss_v2.cpp](examples/test_aclnn_smooth_l1_loss_v2.cpp) | Invokes the SmoothL1Loss operator through the [aclnnsmoothL1Loss](docs/aclnnSmoothL1Loss_en.md) interface. |
