# ForeachSign

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Compute the sign value of each tensor in the input tensor list.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = \left\{
  \begin{aligned}
  1,\quad x_i > 0\\
  0,\quad x_i = 0\\
  -1,\quad x_i < 0
  \end{aligned}
  \right. \quad {,(i=0,1,...n-1)}
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
      <td>Input tensor list for the sign value extraction operation, corresponding to `x` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, INT8, INT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list for the sign value extraction operation, corresponding to `y` in the formula. All tensors in this parameter have the same data type. The data type and data format are the same as those of input parameter `x`, and the shape size equals the shape size of input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, INT32, INT8, INT64, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The number of tensors in x is the same as that in out, and the shapes of tensors at corresponding positions are exactly equal.
- All tensors within a single tensor list of x or out have the same data type, and the data types of x and out are the same. They must be within the supported range declared in the parameter table.
- A single tensor has no more than 8 dimensions.
- A single tensor list in x/y contains no more than 256 tensors.
- When the number of x and out is 0 (empty lists), the API directly returns success without performing computation.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_sign](examples/test_aclnn_foreach_sign.cpp) | Invoke the ForeachSign operator through the [aclnnForeachSign](docs/aclnnForeachSign_en.md) API. |
| Graph mode | -  | Invoke the ForeachSign operator through graph construction using the [operator IR](op_graph/foreach_sign_proto.h).         |
