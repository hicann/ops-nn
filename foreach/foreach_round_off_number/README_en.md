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
      <td>FLOAT, FLOAT16, BFLOAT16, INT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>roundMode</td>
      <td>Input</td>
      <td>Represents the input tensor for the rounding computation, corresponding to `roundMode` in the formulas. The number of elements is 1. The values of roundMode and the corresponding rounding strategies are as follows: when roundMode=0, no rounding operation is performed on the input; when roundMode=1, the input is rounded half to even; when roundMode=2, the input is rounded toward negative infinity; when roundMode=3, the input is rounded toward positive infinity; when roundMode=4, the input is rounded half up; when roundMode=5, the input is rounded toward zero; when roundMode=6, the input is rounded to nearest odd; when roundMode=7, the fractional part of the input is taken; for other values of roundMode, no rounding operation is performed on the input.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list for the rounding operation, corresponding to `y` in the formulas. The data types of all tensors in this parameter are consistent. The data type and data format are the same as those of the input `x`, and the shape size is equal to the shape size of the input `x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 and INT16 are not supported.

## Constraints

- Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.
- Non-contiguous output tensors are automatically converted to contiguous tensors by the framework before computation.
- The tensor list length of x and out cannot exceed 256.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_round_off_number](examples/test_aclnn_foreach_round_off_number.cpp) | Invokes the ForeachRoundOffNumber operator through the [aclnnForeachRoundOffNumber](docs/aclnnForeachRoundOffNumber_en.md) interface. |
| aclnn invocation  | [test_aclnn_foreach_round_off_number_v2](examples/test_aclnn_foreach_round_off_number_v2.cpp) | Invokes the ForeachRoundOffNumberV2 operator through the [aclnnForeachRoundOffNumberV2](docs/aclnnForeachRoundOffNumberV2_en.md) interface. |
| Graph-mode invocation | - | Builds the graph and invokes the ForeachRoundOffNumber operator through the [operator IR](op_graph/foreach_round_off_number_proto.h).         |
