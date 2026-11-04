# HardShrink

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: completes the HardShrink activation function computation. It sets the elements in the input tensor whose absolute values are less than or equal to the threshold lambd to zero, and keeps the elements greater than the threshold unchanged.

- Formula:

$$
\text{HardShrink}(x) = \begin{cases} x, & \text{if } x > \lambda \\ x, & \text{if } x < -\lambda \\ 0, & \text{otherwise} \end{cases}
$$

where x is the input tensor, and $\lambda$ is the threshold parameter lambd.

## Parameters

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input tensor, corresponding to x in the formula. It supports 0-8 dimensions and empty tensors.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>Input</td>
      <td>Threshold parameter, corresponding to λ in the formula. It is a scalar of the float type, with the default value 0.5.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, with the same shape and dtype as self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of self and out must be consistent, supporting FLOAT, FLOAT16, and BFLOAT16.
- The shapes of self and out must be consistent, and broadcasting is not involved.
- self supports 0-8 dimensions.
- self supports empty tensors (with 0 elements). In this case, out is also an empty tensor, and the computation is not executed.
- lambd is a scalar of the float type, and its value has no restriction.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_hard_shrink](./examples/arch35/test_aclnn_hard_shrink.cpp) | Invokes the Hardshrink operator through the [aclnnHardshrink](./docs/aclnnHardshrink_en.md) interface.    |
