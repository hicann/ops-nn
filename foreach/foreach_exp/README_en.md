# ForeachExp

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

- Operator function: Performs the exponential operation on each tensor of the input tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {y_i}=e^{x_i} (i=0,1,...n-1)
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
      <td>Represents the input tensor list for the exponential operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are the same.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list for the exponential operation, corresponding to `y` in the formula. The data types of all tensors in this parameter are the same. When the data type of `x` is INT16, INT8, or UINT8, the data type is FLOAT32; in other cases, it is consistent with the data type of input parameter `x`. The data format is consistent with the data format of input parameter `x`, and the shape is the same as that of input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16, INT16, INT8, and UINT8 are not supported.

## Constraints

- The number of tensors in a single tensor list in x/y cannot exceed 256.
- The number of tensors of x and y must be the same.
- The data types of all tensors in x must be the same.
- The data type of the tensor at the corresponding position in y: FLOAT32 when x is INT16, INT8, or UINT8, and consistent with x in other cases.
- The number of dimensions of a single tensor in x/y ranges from 0 to 8; an error is reported when it is exceeded.
- The shape of the tensor at the corresponding position in y must be consistent with x.
- Empty tensors (with 0 elements) are supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_exp](examples/test_aclnn_foreach_exp.cpp) | Calls the ForeachExp operator through the [aclnnForeachExp](docs/aclnnForeachExp_en.md) interface. |
| GE graph mode | -  | Calls the ForeachExp operator by building a graph through the [operator IR](op_graph/foreach_exp_proto.h).         |
