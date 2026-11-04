# SoftmaxGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Description: completes the backpropagation of [softmax](../softmax_v2/README.md).

- Formula: for the derivative of the Softmax function, you can use the following formula:
  The relationship between gradOutput (the output gradient of the previous layer) and output (the forward output of Softmax) can be expressed as follows:

  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output
  $$

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 813px"><colgroup>
  <col style="width: 103px">
  <col style="width: 144px">
  <col style="width: 194px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Gradient value of the backpropagation, that is, the output gradient of the previous layer. It is gradOutput in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Input</td>
      <td>Output value of the Softmax function, that is, output in the formula.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Input</td>
      <td>Dimension of the Softmax function.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>Output</td>
      <td>The output of the function is the gradient value of the input.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    </tbody>
     </table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: the data types support FLOAT16 and FLOAT32.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_softmax_grad](./examples/test_aclnn_softmax_grad.cpp) | Calls the SoftmaxGrad operator through the [aclnnSoftmaxBackward](./docs/aclnnSoftmaxBackward_en.md) interface. |
