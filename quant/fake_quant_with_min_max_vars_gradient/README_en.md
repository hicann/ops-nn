# FakeQuantWithMinMaxVarsGradient

## Product Support

| Product | Supported |
| :--- | :------: |
| <term>Ascend 950PR&950DT series products</term> | √ |
| <term>Atlas A3 series products</term> | × |
| <term>Atlas A2 series products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference series products</term> | × |
| <term>Atlas training series products</term> | × |

## Function Description

- Operator function: FakeQuantWithMinMaxVarsGradient is the backward gradient operator of FakeQuantWithMinMaxVars, used for the backward pass of quantization-aware training (QAT). It constructs a 0/1 mask from the nudged `nudgedMin`/`nudgedMax`, applies multiplicative gating to gradients (Straight-Through Estimator), and computes the sum of out-of-range gradients for `min`/`max`.

- Computation formulas:

  **Nudge pre-computation (Device side, same as forward FakeQuantWithMinMaxVars):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $nudgedZeroPoint = round(qMin - min / scale)$, clipped to $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  Note: the Nudge intermediate values above are computed on the Kernel Device side from the `min`/`max` tensor inputs; only the `numBits` and `narrowRange` attributes are passed to the kernel.

  **Gradient computation (kernel side):**

  $$mask = \begin{cases} 1, & nudgedMin \le x \le nudgedMax \\ 0, & \text{otherwise} \end{cases}$$

  $$backprops\_wrt\_x = gradients \times mask$$

  $$backprops\_wrt\_min = \sum_{i: x_i < nudgedMin} gradients_i$$

  $$backprops\_wrt\_max = \sum_{i: x_i > nudgedMax} gradients_i$$

  - When `x` is NaN, `mask = 0` (NaN comparison yields false); when `gradients` is NaN, NaN propagates naturally through multiplication (IEEE 754: NaN × 0/1 = NaN).
  - After multiplication, a sign-bit OR is applied to all elements (`vOut_u32 |= vG_u32 & 0x80000000`) to fix a potential -0 sign-bit loss in Ascend Mul; this operation is idempotent and has no side effect on normal values and NaN.
  - When `|min| + |max| < 1e-10` (degenerate): `backprops_wrt_x = gradients` (full passthrough), `backprops_wrt_min = 0`, `backprops_wrt_max = 0`.

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
      <th>Parameter</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data Type</th>
      <th>Data Format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradients</td>
      <td>Input</td>
      <td><ul><li>Upstream gradient tensor, corresponding to gradients in the formulas;</li><li>shape is fully identical to input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Forward input tensor used to build the mask to determine whether x is within the quantization range;</li><li>shape is fully identical to gradients.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Input</td>
      <td><ul><li>Minimum value of the quantization range (scalar tensor);</li><li>shape is rank-0 scalar or (1,);</li><li>acts as a trainable variable in gradient updates.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Input</td>
      <td><ul><li>Maximum value of the quantization range (scalar tensor);</li><li>shape is rank-0 scalar or (1,);</li><li>acts as a trainable variable in gradient updates.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_x</td>
      <td>Output</td>
      <td><ul><li>Gradient of x, the mask-gated gradients;</li><li>shape is identical to input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_min</td>
      <td>Output</td>
      <td><ul><li>Gradient of min, i.e. the sum of gradients at positions where x &lt; nudgedMin;</li><li>shape follows the min/max inputs (rank-0 scalar or (1,)).</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops_wrt_max</td>
      <td>Output</td>
      <td><ul><li>Gradient of max, i.e. the sum of gradients at positions where x &gt; nudgedMax;</li><li>shape follows the min/max inputs (rank-0 scalar or (1,)).</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>Optional attribute</td>
      <td><ul><li>Quantization bit width;</li><li>value range [2, 16];</li><li>default value 8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether to use the narrow quantization range;</li><li>when true, qMin=1; when false, qMin=0;</li><li>default value false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- Inputs `gradients` and `x` support only FLOAT32; their shapes must be fully identical (rank and every dimension size), supporting rank 0~8.
- Inputs `min` and `max` are FLOAT32 with shape rank-0 scalar or (1,); `backprops_wrt_min`/`backprops_wrt_max` output shapes follow the `min`/`max` inputs.
- `num_bits` value range is [2, 16].
- Empty inputs (numel=0) are supported: `backprops_wrt_x` is an empty tensor, `backprops_wrt_min`/`backprops_wrt_max` output 0.
- When `|min| + |max| < 1e-10` the degenerate path is taken: `backprops_wrt_x` directly passes through `gradients`, and `backprops_wrt_min` and `backprops_wrt_max` output 0.

## Invocation

| Invocation Method | Sample Code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_fake_quant_with_min_max_vars_gradient](./examples/test_geir_fake_quant_with_min_max_vars_gradient.cpp) | Invoke the FakeQuantWithMinMaxVarsGradient operator by constructing a graph with GE IR ([operator IR definition](op_graph/fake_quant_with_min_max_vars_gradient_proto.h)). min/max are scalar tensor inputs (trainable variables); num_bits/narrow_range are attributes. TensorFlow path: the TF-native operator `FakeQuantWithMinMaxVarsGradient` is accessed via the TF plugin (name-level mapping), and min/max support rank-0 scalars.|
