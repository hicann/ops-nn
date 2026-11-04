# ForeachNeg

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

- Description: Computes the negation of each tensor in the input tensor list.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = -x_i (i=0,1,...n-1)
  $$

  When the input data type is an integer type, the operator result is consistent with `torch._foreach_neg`, with wrapping performed according to the bit width of the output data type. For example, when the `INT16` input is `-32768`, the output is still `-32768`; when the `INT8` input is `-128`, the output is still `-128`. For `UINT8`, negation is computed with wrapping modulo `256` as an 8-bit unsigned integer, that is, `y = (256 - x) % 256`. For example, when the input is `1`, the output is `255`; when the input is `0`, the output is still `0`.

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
      <td>Denotes the input tensor list for negation computation, corresponding to `x` in the formula. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the output tensor list for negation computation, corresponding to `y` in the formula. The data type and data format are consistent with those of the input parameter `x`, and the shape size equals the shape size of the input parameter `x`. The data types of all tensors in this parameter are consistent.</td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16, INT16, INT8, UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Ascend 950PR&950DT products: INT16, INT8, and UINT8 are not supported.
- <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: BFLOAT16, INT16, INT8, and UINT8 are not supported.

## Constraints

- The number of tensors in x is consistent with that in out, and the shapes of the tensors at corresponding positions are exactly equal.
- The data types of all tensors within a single tensor list of x or out are consistent, and the data types of x and out are consistent; all of them must be within the supported range declared in the parameter table.
- A single tensor has no more than 8 dimensions.
- The number of tensors in a single tensor list of x/y cannot exceed 256.
- When the numbers of x and out are 0 (empty lists), the operator directly returns success without executing the computation.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_foreach_neg](examples/test_aclnn_foreach_neg.cpp) | Invokes the ForeachNeg operator through the [aclnnForeachNeg](docs/aclnnForeachNeg_en.md) interface. |
| Graph mode | -  | Invokes the ForeachNeg operator by building a graph through the [operator IR](op_graph/foreach_neg_proto.h).         |
