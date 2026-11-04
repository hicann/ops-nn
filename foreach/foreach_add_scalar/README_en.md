# ForeachAddScalar

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

- Operator function: Adds the specified tensor value to each tensor in the tensor list.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x}_{i}+{scalar} (i=0,1,...n-1)
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
      <td>Represents the input tensor list of the addition operation, which corresponds to `x` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scalar</td>
      <td>Input</td>
      <td>Represents the input tensor of the addition operation, which corresponds to `scalar` in the formula. The number of elements is 1.</td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the addition operation, which corresponds to `y` in the formula. The data type and data format are the same as the data type and data format of the input parameter `x`, and the shape size is greater than or equal to the shape size of the input parameter `x`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  - When the data type of `x` is FLOAT32, FLOAT16, or INT32, the data type of `scalar` is the same as the data type of `x`.
  - When the data type of `x` is BFLOAT16, the data type of `scalar` supports FLOAT32.
  - When the data type of `x` is INT16, INT8, or UINT8, the data type supports INT32.
- <term>Ascend 950PR&950DT products</term>:
  - INT16, INT8, and UINT8 are not supported.
  - When the data type of `x` is FLOAT32 or INT32, the data type of `scalar` is the same as the data type of `x`.
  - When the data type of `x` is BFLOAT16, the data type of `scalar` supports FLOAT32.
  - When the data type of `x` is FLOAT16, the data type of `scalar` supports FLOAT16, FLOAT32.

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16, INT16, INT8, and UINT8 are not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_add_scalar](examples/test_aclnn_foreach_add_scalar.cpp) | Calls the ForeachAddScalar operator through the [aclnnForeachAddScalar](docs/aclnnForeachAddScalar_en.md) API. |
| aclnn API  | [test_aclnn_foreach_add_scalar_v2](examples/test_aclnn_foreach_add_scalar_v2.cpp) | Calls the ForeachAddScalar operator through the [aclnnForeachAddScalarV2](docs/aclnnForeachAddScalarV2_en.md) API. |
| Graph mode | -  | Calls the ForeachAddScalar operator through graph construction with the [operator IR](op_graph/foreach_add_scalar_proto.h).         |
