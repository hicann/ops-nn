# MseLossV2

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

- Operator function:

  Computes the mean squared error between each element of the input x and the target y.

- Formula:

  When `reduction` is `none`:

$$
  \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
  l_n = \left( x_n - y_n \right)^2,
$$

  Here $x$ is `self`, $y$ is `target`, and $N$ is the size of `batch`. If `reduction` is not `none`, then:

$$
  \ell(x, y) =
  \begin{cases}
      \operatorname{mean}(L), &  \text{if reduction} = \text{'mean';}\\
      \operatorname{sum}(L),  &  \text{if reduction} = \text{'sum'.}
  \end{cases}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 266px">
  <col style="width: 60px">
  <col style="width: 40px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input x in the formula, of any shape as long as it is the same as target. </td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>Input y in the formula, with the same shape as the input.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Input</td>
      <td>Input reduction in the formula, specifying the reduction to be applied to the output, supporting 0('none') | 1('mean') | 2('sum'). 'none' means that no reduction is applied, 'mean' means that the sum of the output is divided by the number of elements in the output, and 'sum' means that the output is summed.</td>
      <td>int64_t</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output <em>l</em>(x,y) in the formula. When the value of reduction is 0, out is consistent with the shape of the tensor obtained by broadcasting self and target; when the value of reduction is 1 or 2, out is a 0-dimensional tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_mse_loss_v2](./examples/test_aclnn_mse_loss_v2.cpp) | Invokes the mse_loss_v2 operator through the [aclnnMseLoss](../mse_loss/docs/aclnnMseLoss_en.md) interface.    |
