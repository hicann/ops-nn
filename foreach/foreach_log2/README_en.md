# ForeachLog2

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

- Operator function: performs the base-2 logarithm function operation on each element in the tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = {\rm log}_2(x_i) (i=0,1,...n-1)
  $$

- Input value range and special value semantics: the input should be a positive number. For a floating-point input element `x`:
  - `x = 0`: the output is negative infinity (-inf).
  - `x < 0`: the output is NaN.
  - `x = +inf`: the output is positive infinity (+inf).
  - `x = -inf`: the output is NaN.
  - `x = NaN`: the output is NaN (passed through).
  - `x > 0` (a normal positive number): the output is log2(x).

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
      <td>Input tensor list for the element-wise base-2 logarithm operation, corresponding to `x` in the formula. The data types of all tensors in this parameter stay consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list for the element-wise base-2 logarithm operation, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, and the shape size is equal to the shape size of the input parameter `x`. The data types of all tensors in this parameter stay consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The output does not support non-contiguous tensors.
- x and out have the same number of tensors, and the tensors at corresponding positions have exactly identical shapes.
- All tensors in a single tensor list of x or out share the same data type, and x and out share the same data type, both of which must be within the supported range declared in the parameter table.
- The dimensionality of a single tensor does not exceed 8.
- The number of tensors in a single tensor list of x/y cannot exceed 256.
- When the number of x and out is 0 (empty list), it returns success directly without performing any computation.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_log2](examples/test_aclnn_foreach_log2.cpp) | Calls the ForeachLog2 operator through the [aclnnForeachLog2](docs/aclnnForeachLog2_en.md) interface. |
| GE graph mode | -  | Calls the ForeachLog2 operator by constructing a graph through the [operator IR](op_graph/foreach_log2_proto.h).         |
