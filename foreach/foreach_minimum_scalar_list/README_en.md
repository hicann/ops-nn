# ForeachMinimumScalarList

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

- Operator function: performs element-wise comparison between the tensor list x and the scalar list scalars, and computes the minimum value corresponding to each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \min(x_i, \text{scalars}_i) (i=0,1,...n-1)
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
      <td>Indicates the input tensor list of the minimum value operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Indicates the input scalar list of the minimum value operation, corresponding to `scalars` in the formula. The number of its elements is equal to the number of tensors in `x`. Its data type has a certain correspondence with the data type of the input parameter: when the data type of input parameter `x` is FLOAT32, FLOAT16, or BFLOAT16, the data type of `scalars` supports FLOAT32; when the data type of input parameter `x` is INT32, the data type of `scalars` supports only INT64.</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Indicates the output tensor list of the minimum value operation, corresponding to `y` in the formula. Its data type and data format are consistent with the data type and data format of input parameter `x`, and its shape size is greater than or equal to the shape size of input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
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
| aclnn interface  | [test_aclnn_foreach_minimum_scalar_list](examples/test_aclnn_foreach_minimum_scalar_list.cpp) | Invokes the ForeachMinimumScalarList operator through the [aclnnForeachMinimumScalarList](docs/aclnnForeachMinimumScalarList_en.md) interface. |
| Graph mode | -  | Invokes the ForeachMinimumScalarList operator through the [operator IR](op_graph/foreach_minimum_scalar_list_proto.h) graph construction method.         |
