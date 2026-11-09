# ForeachExpm1

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

- Operator function: Performs the exponential operation on each tensor in the input tensor list, and then subtracts 1 from the obtained result.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {y_i}=e^{x_i}-1 (i=0,1,...n-1)
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
      <td>Input tensor list for the exponential operation followed by the subtraction-of-1 operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list for the exponential operation followed by the subtraction-of-1 operation, corresponding to `y` in the formula. The data types of all tensors in this parameter are consistent. When the data type of `x` is INT16, INT8, or UINT8, the data type is FLOAT32; in other cases, the data type is the same as that of the input parameter `x`. The data format is the same as that of the input parameter `x`; `y` contains the same number of tensors as `x`, and the shape of `y[i]` is the same as that of `x[i]`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16, INT16, INT8, and UINT8 are not supported.

## Constraints

- In the current implementation on <term>Ascend 950PR&950DT products</term>, the input tensor list `x` can contain at most 256 tensors due to the tiling data capacity. If this limit is exceeded, tiling fails and the operator is not executed.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_expm1](examples/test_aclnn_foreach_expm1.cpp) | Invoke the ForeachExpm1 operator through the [aclnnForeachExpm1](docs/aclnnForeachExpm1_en.md) API. |
| Graph mode | -  | Invoke the ForeachExpm1 operator by building a graph through the [operator IR](op_graph/foreach_expm1_proto.h).         |
