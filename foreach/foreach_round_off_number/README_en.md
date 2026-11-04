# ForeachRoundOffNumber

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

- Operator function: Performs the rounding operation with the specified precision on each tensor in the input tensor list; the rounding method can be specified through roundMode.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = round(x_i, roundMode) (i=0,1,...n-1)
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
      <td>Represents the input tensor list for the rounding operation, corresponding to `x` in the formulas. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roundMode</td>
      <td>Input</td>
      <td>Represents the input tensor for the rounding computation, corresponding to `roundMode` in the formulas. The number of elements is 1. The values of roundMode and the corresponding rounding strategies are as follows: when roundMode=1, the operator performs banker's rounding (round half to even) on the input; when roundMode=2, the operator rounds the input toward negative infinity; when roundMode=3, the operator rounds the input toward positive infinity; when roundMode=4, the operator performs round-half-up rounding on the input; when roundMode=5, the operator rounds the input toward zero; when roundMode=6, the operator performs nearest-odd rounding on the input; for other values of roundMode, the operator performs banker's rounding if precision loss occurs, and does not perform rounding if no precision loss occurs.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list for the rounding operation, corresponding to `y` in the formulas. The data types of all tensors in this parameter are consistent. The data type and data format are the same as those of the input `x`, and the shape size is greater than or equal to the shape size of the input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 and INT16 are not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_round_off_number](examples/test_aclnn_foreach_round_off_number.cpp) | Invokes the ForeachRoundOffNumber operator through the [aclnnForeachRoundOffNumber](docs/aclnnForeachRoundOffNumber_en.md) interface. |
| aclnn invocation  | [test_aclnn_foreach_round_off_number_v2](examples/test_aclnn_foreach_round_off_number_v2.cpp) | Invokes the ForeachRoundOffNumberV2 operator through the [aclnnForeachRoundOffNumberV2](docs/aclnnForeachRoundOffNumberV2_en.md) interface. |
| Graph-mode invocation | - | Builds the graph and invokes the ForeachRoundOffNumber operator through the [operator IR](op_graph/foreach_round_off_number_proto.h).         |
