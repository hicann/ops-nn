# ForeachMulScalarList

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

- Operator function: performs the element-wise multiplication on the input tensor list and the scalar list.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x_i * scalars_i (i=0,1,...n-1)
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
      <td>Input tensor list of the multiplication, corresponding to `x` in the formula. The data types of all tensors in this parameter stay consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Input scalar list of the multiplication, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x`.</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list of x multiplied by scalars, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, and the shape size is greater than or equal to the shape size of the input parameter `x`. The data types of all tensors in this parameter stay consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - When the data type of the input parameter `x` is FLOAT32, FLOAT16, or BFLOAT16, the data type of `scalars` supports only FLOAT32.
  - When the data type of the input parameter `x` is INT32, the data type of `scalars` supports only INT64.
- <term>Ascend 950PR&950DT products</term>: the length of the ScalarList of `scalars` must equal the length of the TensorList of `x`, and the data type has a certain correspondence with the data type of the input parameter `x`:
  - When the data type of `x` is BFLOAT16, FLOAT32, or FLOAT16, the data type supports FLOAT32.
  - When the data type of `x` is INT32, the data type supports INT64.
  - The maximum number of tensors that `x` can contain is 50.

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_mul_scalar_list](examples/test_aclnn_foreach_mul_scalar_list.cpp) | Calls the ForeachMulScalarList operator through the [aclnnForeachMulScalarList](docs/aclnnForeachMulScalarList_en.md) interface. |
| Graph mode | -  | Calls the ForeachMulScalarList operator by constructing a graph through the [operator IR](op_graph/foreach_mul_scalar_list_proto.h).         |
