# FakeQuantWithMinMaxVarsPerChannel

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: performs per-channel fake quantization on the input `x`. Each channel along the last axis (the channel axis) independently performs the nudge quantize-dequantize operation using its own `min[c]`/`max[c]`. The difference from the per-tensor version (FakeQuantWithMinMaxVars) is that each channel has independent quantization parameters, which better adapts to the differences in the value ranges of different channels and significantly improves the weight quantization accuracy.

- Formula:

  **Per-channel nudge precomputation (kernel side, each channel independent):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale[c] = (max[c] - min[c]) / (qMax - qMin)$
  - $scaleInv[c] = (qMax - qMin) / (max[c] - min[c])$ (it must be computed by an independent division, not by $1/scale$, to guarantee precision)
  - $nudgedZeroPoint[c] = round(qMin - min[c] / scale[c])$, clamped to $[qMin, qMax]$
  - $nudgedMin[c] = (qMin - nudgedZeroPoint[c]) \times scale[c]$
  - $nudgedMax[c] = (qMax - nudgedZeroPoint[c]) \times scale[c]$

  **Forward computation (kernel side, per-channel element-wise):**

  $$
  clamped = clamp(x[..., c], nudgedMin[c], nudgedMax[c])
  $$

  $$
  y[..., c] = floor((clamped - nudgedMin[c]) \times scaleInv[c] + 0.5) \times scale[c] + nudgedMin[c]
  $$

  - When a channel satisfies $|min[c]| + |max[c]| = 0$, the output of that channel is all zeros (degenerate processing).
  - When the input `x` is NaN, the NaN propagates to the output `y` after clamp/round.

## Parameters

<table style="table-layout: fixed; width: 1576px"><colgroup>
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
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Denotes the input tensor of the operator, whose last axis is the channel C;</li><li>the rank range is [1, 8];</li><li>the shape is consistent with the output y.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Input</td>
      <td><ul><li>Denotes the minimum value of the per-channel quantization range;</li><li>a 1-D tensor whose length equals the last dimension C of x;</li><li>participates in gradient updates as a trainable variable.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td><ul><li>Denotes the maximum value of the per-channel quantization range;</li><li>a 1-D tensor whose length equals the last dimension C of x;</li><li>participates in gradient updates as a trainable variable.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the quantization bit width;</li><li>the value range is [2, 16];</li><li>the default value is 8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes whether to use the narrow quantization range;</li><li>qMin=1 when true, and qMin=0 when false;</li><li>the default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Denotes the computation output of per-channel fake quantization;</li><li>the shape is exactly the same as the input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types of the input `x` and the output `y` support only FLOAT32, and their shapes must be exactly the same.
- The inputs `min` and `max` must be 1-D tensors (rank=1), and their length must equal the size C of the last dimension of `x`.
- The lengths of `min` and `max` must be consistent.
- The value range of `num_bits` is [2, 16].
- The rank range of `x` must be [1, 8].
- When a channel satisfies `|min[c]| + |max[c]| = 0`, the output of that channel is all zeros.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | - | Invokes the FakeQuantWithMinMaxVarsPerChannel operator by constructing the graph through the GE IR ([operator IR definition](op_graph/fake_quant_with_min_max_vars_per_channel_proto.h)). min/max are 1-D tensor inputs (per-channel trainable variables), and num_bits/narrow_range are attributes. The TF plugin sets the input/output format to NHWC so that the last axis corresponds to the channel.|
