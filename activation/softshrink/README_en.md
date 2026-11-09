# Softshrink

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

> Notes:
> - Ascend 950PR/950DT is provided by the Ascend C implementation in this repository; Atlas A3/A2 and Atlas inference/training products are provided by the canndev TBE implementation. Atlas 200I/500 A2 has no corresponding binary configuration and is not supported.
> - Ascend 950PR/950DT, Atlas A3, and Atlas A2 support FLOAT, FLOAT16, and BFLOAT16. Atlas inference/training products support only FLOAT and FLOAT16, not BFLOAT16.

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
      <td>input_x</td>
      <td>Input</td>
      <td>Input tensor for the Softshrink computation, x in the formula.</td>
      <td>fp16, fp32, bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>Output</td>
      <td>Output tensor of the Softshrink computation. Its shape is the same as that of input_x.</td>
      <td>fp16, fp32, bf16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>Attribute</td>
      <td>Threshold parameter of Softshrink, λ in the formula. Default value: 0.5. NaN and negative values are rejected as invalid parameters; +Inf is allowed.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Only the ND data format is supported.
- The dtype of input_x is fp16/fp32/bf16, and output_y has the same dtype as input_x. Atlas inference/training products do not support BF16 and support only FLOAT and FLOAT16.
- Inputs of any rank are supported (0D to 8D). A 0D scalar input is processed as a 1D tensor with shape (1,).
- lambd must be a FLOAT scalar in the range [0, +∞): 0, positive values, and +Inf are allowed; negative values and NaN return an invalid-parameter error.
- In the Ascend C implementation in this repository, when input_x is NaN, the output is 0. NaN is not propagated; callers that rely on NaN propagation should note this difference.
- Empty tensors (with 0 elements) are supported, and an empty output of the same shape is directly returned.
- Dynamic shapes (including -1 dimensions) are supported: tiling computes the split based on the actual shape at runtime, without recompilation.
- Deterministic implementation by default: the same input always produces the same output.

## Invocation Description

| Invocation method | Sample code | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn API | [test_aclnn_softshrink.cpp](./examples/test_aclnn_softshrink.cpp) | Invoke the Softshrink operator through the [aclnnSoftshrink](./docs/aclnnSoftshrink_en.md) API. |
| GE graph mode | [softshrink_proto.h](./op_graph/softshrink_proto.h) | Invoke the Softshrink operator by constructing a graph with the operator IR. |

> For sample compilation and execution, see the [aclnnSoftshrink sample](./docs/aclnnSoftshrink_en.md#example) and the CANN sample compilation and execution guide.
