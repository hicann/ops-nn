# FakeQuantWithMinMaxArgs

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

- Operator function: Performs per-tensor fake quantization on the input `x`. Through the Nudge algorithm, `min`/`max` are adjusted to integer multiples of the quantization step, and then the quantize-dequantize operation is executed.

- Formula:

  **Nudge precomputation (host side):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $scaleInv = (qMax - qMin) / (max - min)$ (must be computed by an independent division and cannot use $1/scale$, for precision assurance)
  - $nudgedZeroPoint = round(qMin - min / scale)$, clipped to $[qMin, qMax]$ (round is round-half-away, consistent with `std::round` in TF Eigen)
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$
  - $quantZero = floor(-nudgedMin \times scaleInv + 0.5)$ (must be recomputed based on nudgedMin and cannot reuse nudgedZeroPoint, for precision assurance)

  **Forward computation (kernel side):**

  $$
  clamped = clamp(x, nudgedMin, nudgedMax)
  $$

  $$
  q = floor((clamped - nudgedMin) \times scaleInv + 0.5); \quad q\_offset = q - quantZero
  $$

  $$
  y = q\_offset \times scale
  $$

  - When the input `x` is NaN, NaN passes through to the output `y` (NaN passthrough).

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
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Tensor input of the operator, corresponding to x in the formula.</li><li>Its shape is consistent with that of the output y.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>min</td>
      <td>Optional attribute</td>
      <td><ul><li>Minimum value of the quantization range, corresponding to min in the formula.</li><li>It must be a finite value (not NaN/Inf) and less than max.</li><li>Default value: -6.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Optional attribute</td>
      <td><ul><li>Maximum value of the quantization range, corresponding to max in the formula.</li><li>It must be a finite value (not NaN/Inf) and greater than min.</li><li>Default value: 6.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_bits</td>
      <td>Optional attribute</td>
      <td><ul><li>Quantization bit width, corresponding to num_bits in the formula.</li><li>Value range: [2, 16].</li><li>Default value: 8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>narrow_range</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to use the narrow quantization range.</li><li>When it is true, qMin=1; when it is false, qMin=0.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Computation output of fake quantization, corresponding to y in the formula.</li><li>Its shape is the same as that of the input x.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types of the input `x` and the output `y` support only FLOAT32, and their shapes must be exactly identical.
- `min` and `max` must be finite values (not NaN/Inf), and `min` must be less than `max`; `max-min` must not overflow to Inf in FLOAT32, nor be so small that `scale` underflows to 0 or `scaleInv` overflows to Inf (an interval that cannot be represented in FLOAT32 will report an error).
- The value range of `num_bits` is [2, 16].

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph-mode invocation | [test_geir_fake_quant_with_min_max_args](examples/test_geir_fake_quant_with_min_max_args.cpp) | Invokes the FakeQuantWithMinMaxArgs operator through graph construction based on [operator IR definitions](op_graph/fake_quant_with_min_max_args_proto.h) (per-tensor min/max/num_bits/narrow_range scalar attributes). |
