# Relu

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: an activation function that returns a tensor with the same shape as the input tensor. In the tensor, when the value is greater than or equal to 0, the value is taken; when the value is less than 0, 0 is taken.
- Formula:

$$
relu(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
$$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>The input parameter for the Relu computation.</td>
      <td>FLOAT, FLOAT16, INT8, INT32, INT64, BFLOAT16</td>
      <td>ND</td>
      </tr>
  <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output after the Relu computation</td>
      <td>FLOAT, FLOAT16, INT8, INT32, INT64, BFLOAT16</td>
      <td>ND</td>
      </tr>
      </tbody>
  </table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_relu](examples/test_aclnn_relu.cpp) | Invoke the Relu operator through the [aclnnRelu&aclnnInplaceRelu](docs/aclnnRelu&aclnnInplaceRelu_en.md) interface. |
| aclnn invocation | [test_aclnn_inplace_relu](examples/test_aclnn_inplace_relu.cpp) | Invoke the Relu operator through the [aclnnRelu&aclnnInplaceRelu](docs/aclnnRelu&aclnnInplaceRelu_en.md) interface. |
| Graph mode | - | Invoke the Relu operator by constructing a graph through the [operator IR](op_graph/relu_proto.h). |
