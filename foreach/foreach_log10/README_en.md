# ForeachLog10

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

- Operator function: Performs the base-10 logarithm operation on each element in the tensor list.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = {\rm log}_{10}(x_i) (i=0,1,...n-1)
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
      <td>Input tensor list for the element-wise base-10 logarithm operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are the same.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor list for the element-wise base-10 logarithm operation, corresponding to `y` in the formula. The data type and data format are the same as those of input parameter `x`, and the shape size is greater than or equal to the shape size of input parameter `x`. The data types of all tensors in this parameter are the same.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The tensor list `x` and the output list `out` must contain the same number of tensors, and a single tensor list contains no more than 256 tensors; an empty TensorList with 0 members is not supported.
- The number of dimensions of each tensor in `x` and `out` does not exceed 8.
- The graph-mode (GE/IR) path supports dynamic Shape/Rank markers: `-1` indicates a dynamic dimension and `-2` indicates a dynamic rank. Host deduction passes the markers through; concretization must be completed before execution, and the dimensions of each tensor must be specific non-negative values (tiling returns a failure for inputs containing negative dimension markers).
- Non-contiguous tensors are not supported for the output.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_log10](examples/test_aclnn_foreach_log10.cpp) | Calls the ForeachLog10 operator through the [aclnnForeachLog10](docs/aclnnForeachLog10_en.md) interface. |
| GE graph mode | -  | Calls the ForeachLog10 operator by building a graph with the [operator IR](op_graph/foreach_log10_proto.h).         |
