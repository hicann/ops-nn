# FakeQuantWithMinMaxVarsPerChannelGradient

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>     |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: FakeQuantWithMinMaxVarsPerChannelGradient is the backward gradient operator of FakeQuantWithMinMaxVarsPerChannel, used for backpropagation in per-channel quantization-aware training (QAT). It builds a 0/1 mask from the per-channel nudged `nudgedMin[c]`/`nudgedMax[c]`, performs multiplicative gating on the gradients (Straight-Through Estimator), and also computes the per-channel sums of out-of-range gradients of `min`/`max`.

- Formula:

  **Per-channel nudge precomputation (on the device side, the same as the forward FakeQuantWithMinMaxVarsPerChannel):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale[c] = (max[c] - min[c]) / (qMax - qMin)$
  - $nudgedZeroPoint[c] = round(qMin - min[c] / scale[c])$, clipped to $[qMin, qMax]$
  - $nudgedMin[c] = (qMin - nudgedZeroPoint[c]) \times scale[c]$
  - $nudgedMax[c] = (qMax - nudgedZeroPoint[c]) \times scale[c]$

  Note: The nudge intermediate values above are computed on the kernel device side independently per channel from the `min`/`max` tensor inputs.

  **Gradient computation (kernel side, per channel):**

  $$mask[..., c] = \begin{cases} 1, & nudgedMin[c] \le x[..., c] \le nudgedMax[c] \\ 0, & \text{otherwise} \end{cases}$$

  $$backprops\_wrt\_x[..., c] = gradients[..., c] \times mask[..., c]$$

  $$backprops\_wrt\_min[c] = \sum_{i: x_i < nudgedMin[c]} gradients_i$$

  $$backprops\_wrt\_max[c] = \sum_{i: x_i > nudgedMax[c]} gradients_i$$

  - When `x` of a channel is NaN, `mask = 0` (the comparison result with NaN is false); when `gradients` is NaN, NaN propagates naturally through multiplication.
  - When `max[c] <= min[c]` for a channel (a degenerate case, covering the zero-width intervals `|min[c]| + |max[c]| = 0` and `min[c] == max[c] != 0`): `backprops_wrt_x` passes `gradients` through, and `backprops_wrt_min/max` is 0. The degenerate pass-through assumes that `x` of that channel is not NaN; if `x` is NaN, `mask = 0` is still obtained (that is, `backprops_wrt_x = 0`) according to the NaN rule in the preceding item.
  - The cross-row reduction of backprops_wrt_min/max uses Kahan compensated summation to reduce floating-point accumulation errors.

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
      <td>gradients</td>
      <td>Input</td>
      <td><ul><li>Denotes the upstream gradient tensor, whose last axis is the channel C;</li><li>The shape is exactly the same as that of input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Denotes the forward input tensor, used to build the per-channel mask;</li><li>The shape is exactly the same as that of gradients.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Input</td>
      <td><ul><li>Denotes the minimum value of the per-channel quantization range;</li><li>It is a 1-D tensor whose length equals the last dimension C of x;</li><li>It participates in gradient updates as a trainable variable.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td><ul><li>Denotes the maximum value of the per-channel quantization range;</li><li>It is a 1-D tensor whose length equals the last dimension C of x;</li><li>It participates in gradient updates as a trainable variable.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_x</td>
      <td>Output</td>
      <td><ul><li>Denotes the gradient of x, that is, gradients gated by the per-channel mask;</li><li>The shape is the same as that of input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_min</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel gradient of min, that is, the sum of gradients corresponding to the elements whose x &lt; nudgedMin[c] in each channel;</li><li>The shape is [C], consistent with min.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_max</td>
      <td>Output</td>
      <td><ul><li>Denotes the per-channel gradient of max, that is, the sum of gradients corresponding to the elements whose x &gt; nudgedMax[c] in each channel;</li><li>The shape is [C], consistent with max.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the quantization bit width;</li><li>The value range is [2, 16];</li><li>The default value is 8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes whether the narrow quantization range is used;</li><li>When true, qMin=1; when false, qMin=0;</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types of inputs `gradients` and `x` support only FLOAT32, and their shapes must be exactly the same (both the rank and the size of each dimension are identical).
- Inputs `min` and `max` must be 1-D tensors (rank=1), and their lengths must equal the size C of the last dimension of `x`.
- The lengths of `min` and `max` must be consistent.
- The value range of `num_bits` is [2, 16].
- The rank of `x` must be in the range [1, 8], and each dimension must be non-empty (`C >= 1` and the total number of elements > 0).
- Dynamic shape is supported: `-1` in the shape denotes an unknown dimension and `-2` denotes an unknown rank, which are determined by the actual inputs at runtime; during inference, no cross-input comparison is performed for unknown dimensions.
- When `max[c] <= min[c]` for a channel (a degenerate case, covering the zero-width intervals `|min[c]| + |max[c]| = 0` and `min[c] == max[c] != 0`), `backprops_wrt_x` of that channel passes `gradients` through, and `backprops_wrt_min/max` is 0.
- Constraint check locations: the cross-input shape relationships (`gradients` and `x` have the same shape, `min`/`max` are rank-1 with a length equal to the last dimension C of `x`, and `min` and `max` have equal lengths) and the rank range [1, 8] of `x` are checked at the Host InferShape stage; the non-empty dimensions of `x` (`C >= 1` and the total number of elements > 0) and the value range of `num_bits` are checked at the Tiling stage (Tiling is executed before the kernel). When any of the above is not satisfied, `GRAPH_FAILED` is returned and invalid inputs are not dispatched to the kernel; the data types are constrained by operator registration (FLOAT32 only).
- The nudge intermediate values (`scale`, `zp`, `nudgedMin`, `nudgedMax`) and the backward gradients are all evaluated in FLOAT32; `scale` is computed strictly as **one-step Sub followed by one-step Div** according to `(max[c]-min[c])/(qMax-qMin)` (consistent with the forward operator `FakeQuantWithMinMaxVarsPerChannel` in the same repository and with the TF reference model), to ensure bitwise consistency with the reference implementation.
- Special values (`min`/`max`/`x`/`gradients` taking `±FLT_MAX`, `±inf`, or subnormal numbers, and so on) are outside the domain of this operator: in this domain, `scale`, `nudgedMin/nudgedMax`, the mask, and the channel sums are jointly determined by FLOAT32 intermediate results (which may be `±inf`/`NaN`) and by device floating-point behavior. This document does not promise their values, and the comparison results for such inputs are not used as the basis for precision judgment.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_fake_quant_with_min_max_vars_per_channel_gradient.cpp](examples/test_geir_fake_quant_with_min_max_vars_per_channel_gradient.cpp) | Invokes the FakeQuantWithMinMaxVarsPerChannelGradient operator through GE IR graph construction ([operator IR definition](op_graph/fake_quant_with_min_max_vars_per_channel_gradient_proto.h)). min/max are 1-D tensor inputs (per-channel trainable variables), and num_bits/narrow_range are attributes. The TF plugin sets the format of **rank-4** tensors to NHWC only; the rank-1 min/max inputs and the rank-1 backprops_wrt_min/backprops_wrt_max outputs remain ND. |
