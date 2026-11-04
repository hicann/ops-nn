# Sigmoid

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

- Operator function: completes the Sigmoid computation on the input Tensor.

- Formula:

$$
out = {\frac{1} {1+{e}^{-input}}}
$$

## Parameters

  <table style="undefined;table-layout: fixed; width: 1393px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <td>The input parameter for the Sigmoid computation, input in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output parameter of the computation.</td>
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
| aclnn invocation | [test_aclnn_sigmoid](examples/test_aclnn_sigmoid.cpp) | Invoke the Sigmoid operator through the [aclnnSigmoid&aclnnInplaceSigmoid](docs/aclnnSigmoid&aclnnInplaceSigmoid_en.md) interface. |
| aclnn invocation | [test_aclnn_inplace_sigmoid](examples/test_aclnn_inplace_sigmoid.cpp) | Invoke the Sigmoid operator through the [aclnnSigmoid&aclnnInplaceSigmoid](docs/aclnnSigmoid&aclnnInplaceSigmoid_en.md) interface. |
| Graph mode | - | Invoke the Sigmoid operator by constructing a graph through the [operator IR](op_graph/sigmoid_proto.h). |
