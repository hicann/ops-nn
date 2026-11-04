# SoftShrinkGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Backward operator of [Softshrink](../softshrink/README.md), which computes the gradient of backpropagation. The operator provides NaN protection: when input_x is NaN, the gradient is passed through directly (aligned with the PyTorch behavior).

- Formula:

  $$
  output\_y =
  \begin{cases}
  input\_grad, & \lvert input\_x \rvert > lambd \text{ or } input\_x \text{ is NaN} \\
  0, & \text{otherwise}
  \end{cases}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>input_grad</td>
      <td>Input</td>
      <td>Gradient output by the previous step during backpropagation, input_grad in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_x</td>
      <td>Input</td>
      <td>Input tensor of forward Softshrink, input_x in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>Output</td>
      <td>Backpropagation gradient output, output_y in the formula.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>Optional attribute</td>
      <td>Threshold parameter of Softshrink. Default value: 0.5. The value must satisfy lambd≥0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data types of input_grad, input_x, and output_y support fp16, fp32, and bf16.
- <term>Atlas 200I/500 A2 inference products</term>, <term>Atlas inference products</term>, <term>Atlas training products</term>: The data types of input_grad, input_x, and output_y support fp16 and fp32.

## Constraints

- GE graph mode: The dtype and shape of input_grad and input_x must be the same.
- aclnn API (aclnnSoftshrinkBackward): The shapes of input_grad (gradOutput) and input_x (self) must satisfy the broadcast relationship, and the shape of gradInput must be the same as the broadcast result. The dtypes of the two inputs can be different (when they are different, they are internally converted to FLOAT for computation).
- 0D to 8D ND formats are supported.

## Invocation Description

| Invocation method | Sample code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_soft_shrink_grad.cpp](./examples/test_aclnn_soft_shrink_grad.cpp) | Invoke the SoftShrinkGrad operator through the [aclnnSoftshrinkBackward](./docs/aclnnSoftshrinkBackward_en.md) API. |
| GE graph mode | - | Construct and execute the computation graph through the [SoftShrinkGrad operator prototype](./op_graph/soft_shrink_grad_proto.h). |
