# ForeachErf

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

- Operator function: computes the error function (also called the Gaussian error function, error function or Gaussian error function) element-wise.
- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {\rm erf}({x_i}) = \frac{2}{\sqrt{\pi}} \int_{0}^{{x_i}} e^{-t^2} dt (i=0,1,...n-1)
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
      <td>The input tensor list for the error function computation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent. Non-contiguous input tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output tensor list for the error function computation, corresponding to `y` in the formula. The data type and data format are consistent with those of input parameter `x`, and the shape is the same as that of the corresponding tensor in `x`. The data types of all tensors in this parameter are consistent. Non-contiguous output tensors are automatically converted to contiguous tensors by the framework before computation.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, BFLOAT16 is not supported.

## Constraints

- The number of tensors in x or y must not exceed 256.
- The number of tensors in x and y must be the same.
- The shape of the tensor at the corresponding position in y must be the same as that in x.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_foreach_erf](examples/test_aclnn_foreach_erf.cpp) | Invokes the ForeachErf operator through the [aclnnForeachErf](docs/aclnnForeachErf_en.md) API. |
| Graph mode | -  | Invokes the ForeachErf operator through graph construction using the [operator IR](op_graph/foreach_erf_proto.h).         |
