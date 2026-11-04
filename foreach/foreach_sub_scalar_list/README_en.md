# ForeachSubScalarList

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

- Operator function: Performs element-wise subtraction between each tensor of the input tensor list and each scalar of the scalar list scalars.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x_i - scalars_i (i=0,1,...n-1)
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
      <td>Represents the first input tensor list of the subtraction, corresponding to `x` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalars</td>
      <td>Input</td>
      <td>Represents the second input scalar list of the subtraction, corresponding to `scalars` in the formula. The number of elements equals the number of tensors in `x`. It has a certain correspondence with the data type of the input parameter: when the data type of input parameter `x` is FLOAT32, FLOAT16, or BFLOAT16, the data type of `scalars` supports only FLOAT32; when the data type of input parameter `x` is INT32, the data type of `scalars` supports only INT64.</td>
      <td>FLOAT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the subtraction, corresponding to `y` in the formula. All tensors in this parameter have the same data type. The data type and data format are the same as the data type and data format of input parameter `x`, and the shape size is greater than or equal to the shape size of input parameter `x`.</td>
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
| aclnn API  | [test_aclnn_foreach_sub_scalar_list](examples/test_aclnn_foreach_sub_scalar_list.cpp) | Invokes the ForeachSubScalarList operator through the [aclnnForeachSubScalarList](docs/aclnnForeachSubScalarList_en.md) interface. |
| GE graph mode | -  | Invokes the ForeachSubScalarList operator through the operator IR-based [graph construction](op_graph/foreach_sub_scalar_list_proto.h) method.         |
