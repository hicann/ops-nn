# ForeachAsin

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
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

- Operator function: performs the arcsine function computation element by element.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \sin^{-1}(x_i) (i=0,1,...n-1)
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
      <td>Denotes the list of input tensors for the arcsine computation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the list of output tensors of the arcsine computation, corresponding to `y` in the formula. The number of tensors is consistent with the input `x`. The data types of all tensors in this parameter are consistent. The data type and data format are consistent with the data type and data format of the input `x`, and the shape is consistent with the tensor at the corresponding position of the input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

- The number of tensors in a single tensor list in x/y cannot exceed 256.
- The number of tensors in x and y must be consistent.
- The data types of all tensors in x must be consistent.
- The data type of the tensor at the corresponding position in y is consistent with x.
- The dimensions of a single tensor in x/y range from 0 to 8 dimensions; an error is reported when the range is exceeded.
- The shape of the tensor at the corresponding position in y must be consistent with x.
- Empty tensors (with an element count of 0) are supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_asin](examples/test_aclnn_foreach_asin.cpp) | Calls the ForeachAsin operator through the [aclnnForeachAsin](docs/aclnnForeachAsin_en.md) interface. |
| GE graph mode | -  | Calls the ForeachAsin operator by building a graph through [operator IR](op_graph/foreach_asin_proto.h).         |
