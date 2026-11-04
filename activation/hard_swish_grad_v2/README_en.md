# HardSwishGradV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Description: Backpropagation of [aclnnHardswish](../hard_swish/docs/aclnnHardswish&aclnnInplaceHardswish_en.md), completing the gradient computation of the tensor self.
- Description difference: Compared with the [aclnnHardswishBackward](../hard_swish_grad/docs/aclnnHardswishBackward_en.md) interface, the calculation formula of aclnnHardswishBackwardV2 has been slightly adjusted. See "Formula" for details.
- Formula:

  $$
  out_{i} = gradOutput_{i} \times gradSelf_{i}
  $$

  The calculation formula of gradSelf is

  $$
  gradSelf_{i} = \begin{cases}
  0, & self_{i} \le -3, \\
  self_{i} / 3 + 0.5, & -3 \lt self_{i} \lt 3, \\
  1, & self_{i} \ge 3
  \end{cases}
  $$

  For non-finite values, the operator performs the computation through comparison, selection, and multiplication: when `self` is NaN, both comparison results are false, and `gradSelf` is set to 1;
  when `gradOutput` is positive or negative Inf and `gradSelf` is 0, the output follows the IEEE 754 multiplication semantics and the result is NaN.

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>gradOutput</td>
      <td>Input</td>
      <td>Empty tensors are supported. Denotes the input tensor, the input gradOutput in the formula. Its shape must be the same as that of self and out, and its data type must be the same as that of self and out.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Empty tensors are supported. Input data. It is self in the formula. Its shape must be the same as that of gradOutput and out, and its data type must be the same as that of gradOutput and out.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Empty tensors are supported. Denotes the output tensor, out in the formula. Its shape must be the same as that of gradOutput and self, and its data type must be the same as that of gradOutput and self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    </tbody>
  </table>

   - <term>Atlas training products</term>: The supported data types are FLOAT16 and FLOAT32.

## Constraints

- The shape dimensions of `gradOutput`, `self`, and `out` do not exceed 8.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_hard_swish_grad_v2](./examples/test_aclnn_hard_swish_backward_v2.cpp) | Calls the HardswishBackwardV2 operator through the [aclnnHardswishBackwardV2](./docs/aclnnHardswishBackwardV2_en.md) interface.    |
| Graph-mode invocation | -   | Calls the HardswishBackwardV2 operator by building a graph through [operator IR](./op_graph/hard_swish_grad_v2_proto.h). |
