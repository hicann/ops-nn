# ForeachMulScalar

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

- Operator function: performs the multiplication operation between each tensor of the input tensor list and the tensor scalar.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x_i * scalar (i=0,1,...n-1)
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
      <td>Indicates the input tensor list of the multiplication operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Indicates the input tensor of the multiplication operation, corresponding to `scalar` in the formula. The number of its elements is 1.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Indicates the output tensor list of x multiplied by scalar, corresponding to `y` in the formula. Its data type and data format are consistent with the data type and data format of input parameter `x`, and its shape size is greater than or equal to the shape size of input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
  - When the data type of `x` is FLOAT32, FLOAT16, or INT32, the data type of `scalar` is consistent with the data type of `x`.
  - When the data type of `x` is BFLOAT16, the data type of `scalar` supports FLOAT32.
- <term>Ascend 950PR&950DT products</term>:
  - When the data type of `x` is FLOAT32 or INT32, the data type of `scalar` is consistent with the data type of `x`.
  - When the data type of `x` is BFLOAT16, the data type of `scalar` supports FLOAT32.
  - When the data type of `x` is FLOAT16, the data type of `scalar` supports FLOAT16 and FLOAT32.
  - The maximum number of tensors contained in `x` is 50.

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_mul_scalar](examples/test_aclnn_foreach_mul_scalar.cpp) | Invokes the ForeachMulScalar operator through the [aclnnForeachMulScalar](docs/aclnnForeachMulScalar_en.md) interface. |
| aclnn interface  | [test_aclnn_foreach_mul_scalar_v2](examples/test_aclnn_foreach_mul_scalar_v2.cpp) | Invokes the ForeachMulScalar operator through the [aclnnForeachMulScalarV2](docs/aclnnForeachMulScalarV2_en.md) interface. |
| Graph mode | -  | Invokes the ForeachMulScalar operator through the [operator IR](op_graph/foreach_mul_scalar_proto.h) graph construction method.         |
