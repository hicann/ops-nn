# SquaredRelu

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>kirin X90 processor series products</term>    |     √    |
|  <term>Kirin 9030 processor series products</term>    |     √    |

## Function

- Description:

  The SquaredRelu function is a variant based on the standard Relu function. Its main feature is squaring the output of the Relu function, and it is often used as the activation function of a model.

- Formula:

  $$
  y_i=(Relu(x_i))^2
  $$

  The formula of Relu is:

  $$
  Relu(x_i)=max(0,x_i)
  $$

  where x is the input tensor, y is the output tensor, and i is the position index of the element in the tensor.

## Parameters

<table style="undefined;table-layout: fixed; width: 773px"><colgroup>
  <col style="width: 91px">
  <col style="width: 154px">
  <col style="width: 156px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- If the input value is NaN, the output is also NaN. If the input is Inf, the output is also Inf.
- If the input is -Inf, the output is 0.
- The input shape supports only 1 to 8 dimensions. An error is reported when the value is out of the range.
- out supports non-contiguous tensors. The computation result is written to the logical view of out.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_squared_relu](./examples/test_aclnn_squared_relu.cpp) | Invoke the SquaredRelu operator through the [aclnnSquaredRelu](./docs/aclnnSquaredRelu_en.md) API.    |
| Graph-mode invocation | -   | Invoke the SquaredRelu operator by building a graph through the [operator IR](./op_graph/squared_relu_proto.h). |
