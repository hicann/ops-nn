# ForeachLog1p

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

- Operator function: Adds 1 first and then applies the logarithm function with base e to each element in the tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = {\rm log}_e(x_i + 1) (i=0,1,...n-1)
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
      <td>Represents the input tensor list for adding 1 first and then applying the logarithm function with base e, corresponding to `x` in the formula. The data types of all tensors in this parameter are the same. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the final output result, corresponding to `y` in the formula. The data type and data format are consistent with those of input parameter `x`, and the shape is the same as that of the corresponding tensor in `x`. The data types of all tensors in this parameter are the same. Non-contiguous output tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- Number of tensors in x/y <= 256.
- The number of tensors in x and y must be the same.
- The shape of the corresponding tensor in y must be the same as that in x.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_log1p](examples/test_aclnn_foreach_log1p.cpp) | Calls the ForeachLog1p operator through the [aclnnForeachLog1p](docs/aclnnForeachLog1p_en.md) interface. |
| Graph-mode invocation | -  | Calls the ForeachLog1p operator by building a graph through the [operator IR](op_graph/foreach_log1p_proto.h).         |
