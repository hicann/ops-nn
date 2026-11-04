# FakeQuantWithMinMaxArgsGradient

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

- Operator function: Backward gradient operator of FakeQuantWithMinMaxArgs. It builds a 0/1 mask with the nudged `nudgedMin`/`nudgedMax` to perform multiplicative gating on the gradient.

- Formula:

  **Nudge precomputation (host side, the same as FakeQuantWithMinMaxArgs):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $nudgedZeroPoint = round(qMin - min / scale)$, clipped to $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  Note: The preceding Nudge intermediate values are used only for host-side precomputation, and only `nudgedMin` and `nudgedMax` are passed to the kernel.

  **Gradient computation (kernel side):**

  $$
  mask = \begin{cases} 1, & nudgedMin \le x \le nudgedMax \\ 0, & \text{otherwise} \end{cases}
  $$

  $$
  y = gradients \times mask
  $$

  - When `x` is NaN, `mask = 0` (the NaN comparison result is false). When `gradients` is NaN, NaN propagates naturally through multiplication (IEEE754: NaN × 0/1 = NaN).
  - After multiplication, sign-bit OR is performed on all elements (vOut_u32 |= vG_u32 & 0x80000000), which fixes the problem that the Ascend Mul may lose the -0 sign bit. For normal values and NaN, this operation is idempotent and has no side effects.

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
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradients</td>
      <td>Input</td>
      <td><ul><li>Upstream gradient tensor, corresponding to gradients in the formula.</li><li>The shape is the same as the input x and the output y.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Forward input tensor, used to build the mask to determine whether x is in the quantization range.</li><li>The shape is the same as gradients and the output y.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Optional attribute</td>
      <td><ul><li>Minimum value of the quantization range.</li><li>It must be smaller than max.</li><li>The default value is -6.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Optional attribute</td>
      <td><ul><li>Maximum value of the quantization range.</li><li>It must be greater than min.</li><li>The default value is 6.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>Optional attribute</td>
      <td><ul><li>Quantization bit width.</li><li>The value range is [2, 16].</li><li>The default value is 8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>Optional attribute</td>
      <td><ul><li>Whether to use the narrow quantization range.</li><li>qMin=1 when true, and qMin=0 when false.</li><li>The default value is false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Gradient computation output, corresponding to y in the formula.</li><li>The shape is the same as the input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types of the input `gradients`, the input `x`, and the output `y` support only FLOAT32, and their shapes must be exactly the same.
- `min` must be smaller than `max`.
- The value range of `num_bits` is [2, 16].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_fake_quant_with_min_max_args_gradient](examples/test_geir_fake_quant_with_min_max_args_gradient.cpp) | Calls the FakeQuantWithMinMaxArgsGradient operator by constructing a graph in GE IR (per-tensor min/max/num_bits/narrow_range scalar attributes). |
