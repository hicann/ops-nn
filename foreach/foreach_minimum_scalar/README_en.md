# ForeachMinimumScalar

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

- Operator function: performs an element-wise comparison between the tensor list x and the tensor scalar, and computes the minimum value corresponding to each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \min(x_i, \text{scalar}) (i=0,1,...n-1)
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
      <td>Denotes the input tensor list of the minimum operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent, and the maximum number of tensors is 256.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Denotes the input tensor of the minimum operation, corresponding to `scalar` in the formula. The number of elements is 1. The data type has a certain correspondence with the data type of the input parameter `x`: when the data type of `x` is FLOAT32, FLOAT16, or INT32, the data type is consistent with the data type of `x`; when the data type of `x` is BFLOAT16, the data type supports FLOAT32.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list of the minimum operation, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, the shape size is the same as the shape size of the input parameter `x`, and the number of tensors is the same as the number of tensors in `x`. The data types of all tensors in this parameter are consistent.</td>
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
| aclnn invocation  | [test_aclnn_foreach_minimum_scalar](examples/test_aclnn_foreach_minimum_scalar.cpp) | Invokes the ForeachMinimumScalar operator through the [aclnnForeachMinimumScalar](docs/aclnnForeachMinimumScalar_en.md) API. |
| aclnn invocation  | [test_aclnn_foreach_minimum_scalar_v2](examples/test_aclnn_foreach_minimum_scalar_v2.cpp) | Invokes the ForeachMinimumScalar operator through the [aclnnForeachMinimumScalarV2](docs/aclnnForeachMinimumScalarV2_en.md) API. |
| Graph mode | -  | Invokes the ForeachMinimumScalar operator by constructing the graph through the [operator IR](op_graph/foreach_minimum_scalar_proto.h).         |
