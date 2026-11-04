# ForeachLogInplace

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
|  <term>Kirin X90 processor series products</term> | × |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: performs the element-wise natural logarithm computation (in place) on each tensor in the input tensor list.
- Formula:

  $$
  x_i = \ln(x_i)
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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Supports empty tensors. Represents the input tensor list for the element-wise logarithm computation with base e, corresponding to `x` in the formula. The value range is (0, +∞). The data types of all tensors in this parameter are consistent, and the number of dimensions (Dim) of each tensor is not greater than 8.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Supports empty tensors. Represents the computation result tensor list. It shares memory with the input `x`, and the computation result is written back in place. The shape and data type of each tensor in this parameter are consistent with those of the tensor at the corresponding position in the input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- A single tensor list of the input parameter `x` contains no more than 256 tensors.

## Invocation Description

| Invocation method | Sample Code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_foreach_log_inplace](examples/arch35/test_aclnn_foreach_log_inplace.cpp) | Invokes the ForeachLogInplace operator through the [aclnnForeachLogInplace](docs/aclnnForeachLogInplace_en.md) API. |
