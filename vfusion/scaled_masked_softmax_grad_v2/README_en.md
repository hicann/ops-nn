# ScaledMaskedSoftmaxBackward

English | [简体中文](./README.md)

## Supported Product Models

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |

## Function

- Operator function: backpropagation of softmax, with scaling and masking applied to the result.
- Formula:

  $$
  out = gradOutput \cdot output - sum(gradOutput \cdot output)\cdot output \\
  out = \begin{cases}
  out * scale, &\text { mask is 0 } \\
  0, &\text { mask is 1 }
  \end{cases}
  $$

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
      <td>Input gradOutput in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Input</td>
      <td>Input y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>Input mask in the formula.</td>
      <td>BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Attribute</td>
      <td>The default value is 1.0.</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fixTriuMask</td>
      <td>Attribute</td>
      <td>Only false is supported</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>out in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_scaled_masked_softmax_backward](./examples/test_aclnn_scaled_masked_softmax_backward.cpp) | Invokes the aclnnScaledMaskedSoftmaxBackward operator through the [aclnnScaledMaskedSoftmaxBackward](./docs/aclnnScaledMaskedSoftmaxBackward_en.md) interface.    |
