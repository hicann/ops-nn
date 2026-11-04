# ForeachDivScalarList

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

- Operator function: performs element-wise division between tensor list x and scalar list scalars.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \frac{x_i}{{\rm scalars}_i} (i=0,1,...n-1)
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
      <td>Denotes the first input tensor list of the division operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Denotes the second input scalar list of the division operation, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x`.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list of the division operation, corresponding to `y` in the formula. The data type and data format are consistent with the data type and data format of input parameter `x`, and the shape size is greater than or equal to the shape size of input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: the data types of x and y do not support BFLOAT16.

## Constraints

- Outputs do not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_div_scalar_list](examples/test_aclnn_foreach_div_scalar_list.cpp) | Invokes the ForeachDivScalarList operator through the [aclnnForeachDivScalarList](docs/aclnnForeachDivScalarList_en.md) interface. |
| Graph-mode invocation | -  | Invokes the ForeachDivScalarList operator through graph construction with the [operator IR](op_graph/foreach_div_scalar_list_proto.h).         |
