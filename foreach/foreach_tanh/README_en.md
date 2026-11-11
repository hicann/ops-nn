# ForeachTanh

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Performs the hyperbolic tangent function computation on each tensor of the input tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i=\frac{e^{x_i}-e^{-x_i}}{e^{x_i}+e^{-x_i}} (i=0,1,...n-1)
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>Denotes the input tensor list on which the hyperbolic tangent function computation is performed, corresponding to `x` in the formula. The data types of all tensors in this parameter are the same.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list on which the hyperbolic tangent function computation is performed, corresponding to `y` in the formula. The data types of all tensors in this parameter are the same. The data type and data format are consistent with the data type and data format of the input parameter `x`, and the shape is the same as that of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The number of tensors in x is the same as that in y, and the shapes of the tensors at corresponding positions are exactly equal.
- The data types of all tensors in a single tensor list of x or y are the same, and the data types of x and y are the same. All of them must be within the support range declared in the parameter table.
- The dimensions of a single tensor do not exceed 8.
- The number of tensors in a single tensor list in x/y cannot exceed 256.
- Empty tensors (with 0 elements) are supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_tanh](examples/test_aclnn_foreach_tanh.cpp) | Calls the ForeachTanh operator through the [aclnnForeachTanh](docs/aclnnForeachTanh_en.md) interface. |
| Graph mode | -  | Calls the ForeachTanh operator by building a graph through [operator IR](op_graph/foreach_tanh_proto.h).         |
