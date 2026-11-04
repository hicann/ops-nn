# ScaledMaskedSoftmaxV2

English | [简体中文](./README.md)

## Supported Product Models

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: first multiplies input x by scale, then sets the positions where the mask is true to -inf, and finally performs softmax along the last dimension. The positions where the mask is false retain the scaled values.
- Implementation description: when the mask is true in the formula, the value is set to $-\infty$; the actual implementation fills in the minimum value of the data type at the corresponding computation precision (dtype min) instead of IEEE $-inf$. The specific value is consistent with the computation path: the FLOAT16 path uses the `float16` minimum value (corresponding bit pattern `0xFBFF`), and the FLOAT and BFLOAT16 paths use the `float32` minimum value (corresponding bit pattern `0xFF7FFFFF`) when computing in float32. After the `exp` of softmax, this value also approaches 0, which is numerically equivalent to setting the value to $-\infty$.
- Formula:

  $$
  x_i' = -\infty, \quad mask_i = true
  $$

  $$
  x_i' = scale \times x_i, \quad mask_i = false
  $$

  $$
  y = Softmax(x', dim = -1)
  $$

  $$
  Softmax(X_i) ={e^{X_i - max(X, dim=-1)} \over \sum{e^{X_i - max(X, dim=-1)}}}
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
      <td>x</td>
      <td>Input</td>
      <td>Input x in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>bool mask. The positions with true are set to -inf before softmax (actually the dtype minimum value, see the implementation description in the Function section), and the positions with false retain scale * x.</td>
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
      <td>fixedTriuMask</td>
      <td>Attribute</td>
      <td>Only false is supported</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>y in the formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_scaled_masked_softmax](./examples/test_aclnn_scaled_masked_softmax.cpp) | Invokes the aclnnScaledMaskedSoftmax operator through the [aclnnScaledMaskedSoftmax](./docs/aclnnScaledMaskedSoftmax_en.md) interface.    |
