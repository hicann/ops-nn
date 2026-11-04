# ForeachDivList

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

- Operator function: performs element-wise division on tensor list x1 and tensor list x2.

- Computation formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \frac{x1_i}{x2_i} (i=0,1,...n-1)
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
      <td>x1</td>
      <td>Input</td>
      <td>Denotes the first input tensor list of the division operation, corresponding to `x1` in the formula. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Denotes the second input tensor list of the division operation, corresponding to `x2` in the formula. Its data type, data format, and shape are consistent with those of input parameter `x1`, and all tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list of the division operation, corresponding to `y` in the formula. Its data type and data format are consistent with those of input parameter `x1`, and its shape is consistent with the shape of input parameter `x1`. All tensors in this parameter have the same data type.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The number of tensors contained in tensor lists `x1` and `x2` must be equal and consistent with the number of elements of `y`, and the number of tensors in a single tensor list does not exceed 256.
- Each tensor in `x1`, `x2`, and `y` has no more than 8 dimensions, and the shapes are the same.
- The `-1` (dynamic dimensions) and `-2` (dynamic rank) markers are processed by the Host deduction stage according to the framework conventions; the kernel-side Tiling requires that the shape has been materialized (the element count is non-negative).
- Non-contiguous output tensor support: aclnnForeachDivList supports non-contiguous output tensors (internally converted to contiguous automatically and written back); graph mode (IR) does not support non-contiguous output tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_div_list](examples/test_aclnn_foreach_div_list.cpp) | Calls the ForeachDivList operator through the [aclnnForeachDivList](docs/aclnnForeachDivList_en.md) interface. |
| GE graph mode | -  | Calls the ForeachDivList operator by constructing a graph through the [operator IR](op_graph/foreach_div_list_proto.h).         |
