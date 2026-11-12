# ForeachMinimumList

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

- Operator function: Performs element-wise comparison on the tensor list x1 and the tensor list x2, and computes the minimum value of each corresponding element.
- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \min(x1_i, x2_i) (i=0,1,...n-1)
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
      <td>Represents the input tensor list of the minimum operation, corresponding to `x1` in the formulas. The shape is consistent with that of the input `x2`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Represents the input tensor list of the minimum operation, corresponding to `x2` in the formulas. The data type, data format, and shape are consistent with those of the input `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the output tensor list of the minimum operation, corresponding to `y` in the formulas. The data type and data format are the same as those of the input `x1`, and the shape size is greater than or equal to the shape size of the input `x1`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_minimum_list](examples/test_aclnn_foreach_minimum_list.cpp) | Invokes the ForeachMinimumList operator through the [aclnnForeachMinimumList](docs/aclnnForeachMinimumList_en.md) interface. |
| GE Graph-mode invocation | [test_geir_foreach_minimum_list](examples/test_geir_foreach_minimum_list.cpp) | Builds the graph and invokes the ForeachMinimumList operator through the [operator IR](op_graph/foreach_minimum_list_proto.h).         |
