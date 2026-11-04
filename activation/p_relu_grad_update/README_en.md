# PReluGradUpdate

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: An activation function that returns a tensor with the same shape as the input tensor. When a value in the tensor is greater than or equal to 0, the value is taken; when it is less than 0, 0 is taken.
- Formula:

  $$
  gradInput
  =
  \begin{cases}
  gradOutput, & self > 0 \\
  weight \times gradOutput, & self \le 0
  \end{cases}
  $$

  $$
  update
  =
  \begin{cases}
  0, & self > 0 \\
  self \times gradOutput, & self \le 0
  \end{cases}
  $$

## Parameters

 <table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>grads</td>
      <td>Input</td>
      <td>Gradient of the backward propagation of the prelu forward output y</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  <tr>
      <td>features</td>
      <td>Input</td>
      <td>Input features of the prelu forward pass</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>weights</td>
      <td>Input</td>
      <td>Input weights of the prelu forward pass</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Gradient of features</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
  </tr>
  <tr>
      <td>update</td>
      <td>Output</td>
      <td>Intermediate result of the weights gradient computation, used for subsequent reduce computation</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
  </tr>
   </tbody>
  </table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_prelu_backward](./examples/test_aclnn_prelu_backward.cpp) | Calls the PReluGradUpdate operator through the [aclnnPReluBackward](./docs/aclnnPreluBackward_en.md) interface. |
