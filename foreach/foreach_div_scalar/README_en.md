# ForeachDivScalar

English | [简体中文](./README.md)

## Applicable Products

| Product                           | Support |
| :---------------------------------------------------------| :--------:|
| <term>Ascend 950PR&950DT products</term>          | √    |
| <term>Atlas A3 products</term> | √    |
| <term>Atlas A2 products</term> | √    |
| <term>Atlas 200I/500 A2 inference products</term>         | ×    |
| <term>Atlas inference products</term>             | ×    |
| <term>Atlas training products</term>             | ×    |
| <term>Kirin X90 processor series products</term>          | √    |
| <term>Kirin 9030 processor series products</term>          | √    |

## Function

- Operator function: computes the tensor list x divided by the tensor scalar.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \frac{x_i}{{\rm scalar}} (i=0,1,...n-1)
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
      <td>Represents the first input tensor list of the division, corresponding to `x` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Represents the second input tensor of the division, corresponding to `scalar` in the formula. The number of elements is 1. The data type has a certain correspondence with the data type of input `x`: when the data type of `x` is FLOAT32 or FLOAT16, the data type is consistent with that of `x` or DOUBLE; when the data type of `x` is BFLOAT16, the supported data types are FLOAT32 and DOUBLE.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of tensor x divided by tensor scalar, corresponding to `y` in the formula. Its data type and data format are consistent with those of input `x`, and its shape size is greater than or equal to the shape size of input `x`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported for the data types of x and y.

## Constraints

- Non-contiguous tensors are not supported for the output.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_div_scalar](examples/test_aclnn_foreach_div_scalar.cpp) | Invokes the ForeachDivScalar operator through the [aclnnForeachDivScalar](docs/aclnnForeachDivScalar_en.md) interface. |
| aclnn interface  | [test_aclnn_foreach_div_scalar_v2](examples/test_aclnn_foreach_div_scalar_v2.cpp) | Invokes the ForeachDivScalar operator through the [aclnnForeachDivScalarV2](docs/aclnnForeachDivScalarV2_en.md) interface. |
| Graph mode | -  | Invokes the ForeachDivScalar operator by building a graph through the [operator IR](op_graph/foreach_div_scalar_proto.h).         |
