# ForeachZeroInplace

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

- Operator function: updates the input tensor list in place, setting each tensor of the input tensor list to 0.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  $$

  $$
  x_i = 0 (i=0,1,...n-1)
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
      <td>Input/Output</td>
      <td>Denotes the input and output tensor list of the set-to-zero operation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent. The data type, data format, and shape of the output parameter `x` are consistent with those of the input parameter `x`.</td>
      <td>FLOAT32, FLOAT16, INT32, INT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The input/output of the inplace operation does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_foreach_zero_inplace](examples/test_aclnn_foreach_zero_inplace.cpp) | Invokes the ForeachZeroInplace operator through the [aclnnForeachZeroInplace](docs/aclnnForeachZeroInplace_en.md) API. |
| Graph mode | -  | Invokes the ForeachZeroInplace operator by constructing the graph through the [operator IR](op_graph/foreach_zero_inplace_proto.h).         |
