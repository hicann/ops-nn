# ForeachPowScalarList

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

- Operator function: Performs the power operation on each tensor in the input tensor list x, with the exponent being the corresponding scalar in the scalar list exponent.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  exponent = [{exponent_0}, {exponent_1}, ... {exponent_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {{x_i}}^{exponent_i} (i=0,1,...n-1)
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
      <td>Represents the base of the power operation, corresponding to `x` in the formulas. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Represents the exponent of the power operation, corresponding to `exponent` in the formulas. The number of elements equals the number of tensors in `x`. The data type has a certain correspondence with the data type of the input `x`: when the data type of x is FLOAT32, FLOAT16, or BFLOAT16, the data type supports FLOAT32; when the data type of x is INT32, the data type supports INT64.</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output result of the power operation, corresponding to `y` in the formulas. The data types of all tensors in this parameter are consistent. The data type and data format are the same as those of the input `x`, and the shape size is greater than or equal to the shape size of the input `x`.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of x and y do not support BFLOAT16.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_pow_scalar_list](examples/test_aclnn_foreach_pow_scalar_list.cpp) | Invokes the ForeachPowScalarList operator through the [aclnnForeachPowScalarList](docs/aclnnForeachPowScalarList_en.md) interface. |
| GE Graph-mode invocation | -  | Builds the graph and invokes the ForeachPowScalarList operator through the [operator IR](op_graph/foreach_pow_scalar_list_proto.h).         |
