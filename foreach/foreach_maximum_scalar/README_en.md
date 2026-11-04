# ForeachMaximumScalar

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

- Operator function: performs an element-wise comparison between the tensor list and the tensor scalar, and computes the maximum value corresponding to each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \max(x_i, \text{scalar}) (i=0,1,...n-1)
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
      <td>Denotes the input tensor list of the maximum operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Denotes the input tensor of the maximum operation, corresponding to `scalar` in the formula. The number of elements is 1. The data type has a certain correspondence with the data type of the input parameter `x`: when the data type of `x` is FLOAT32, FLOAT16, or INT32, the data type is consistent with the data type of `x`; when the data type of `x` is BFLOAT16, the data type supports FLOAT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list of the maximum operation, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, and the shape size is greater than or equal to the shape size of the input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: the data types of x and y do not support BFLOAT16.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_maximum_scalar](examples/test_aclnn_foreach_maximum_scalar.cpp) | Invokes the ForeachMaximumScalar operator through the [aclnnForeachMaximumScalar](docs/aclnnForeachMaximumScalar_en.md) API. |
| aclnn invocation  | [test_aclnn_foreach_maximum_scalar_v2](examples/test_aclnn_foreach_maximum_scalar_v2.cpp) | Invokes the ForeachMaximumScalar operator through the [aclnnForeachMaximumScalarV2](docs/aclnnForeachMaximumScalarV2_en.md) API. |
| Graph mode | -  | Invokes the ForeachMaximumScalar operator by constructing the graph through the [operator IR](op_graph/foreach_maximum_scalar_proto.h).         |
