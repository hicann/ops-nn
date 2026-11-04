# Softshrink

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

- Operator function: Apply the Softshrink activation function to the input tensor.

- Formula:

$$
Softshrink(x)=\begin{cases}
x-λ, if x > λ \\
x+λ, if x < -λ \\
0, otherwise \\
\end{cases}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Input tensor for the Softshrink computation, x in the formula.</td>
      <td>fp16, fp32, bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>Input</td>
      <td>Threshold parameter of Softshrink. Default value: 0.5.
      </td>
      <td>fp32</td>
      <td>1</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output tensor of the Softshrink computation.</td>
      <td>fp16, fp32, bf16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Only the ND data format is supported.
- The dtype of input x is fp16/fp32/bf16, and output y has the same dtype as input x.
- Inputs of any rank are supported (0D to 8D). A 0D scalar input is processed as a 1D tensor with shape (1,).
- The value range of lambd is [0, +∞). Negative values and nan are not allowed.
- Empty tensors (with 0 elements) are supported, and an empty output of the same shape is directly returned.
- Dynamic shapes (including -1 dimensions) are supported: tiling computes the split based on the actual shape at runtime, without recompilation.
- Deterministic implementation by default: the same input always produces the same output.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_softshrink.cpp](./examples/test_aclnn_softshrink.cpp) | Invoke the Softshrink operator through the [aclnnSoftshrink](./docs/aclnnSoftshrink_en.md) API. |
