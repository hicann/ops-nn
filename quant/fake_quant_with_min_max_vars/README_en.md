# FakeQuantWithMinMaxVars

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: performs per-tensor fake quantization on the input `x`. Through the Nudge algorithm, `min`/`max` are adjusted to integer multiples of the quantization step, and then the quantize-dequantize operation is performed. `min`/`max` participate in gradient updates as trainable variables (tensor inputs), which applies to quantization-aware training (QAT). The corresponding backward gradient operator is FakeQuantWithMinMaxVarsGradient.

- Formula:

  **Nudge precomputation (kernel side, computed from the `min`/`max` tensor inputs):**

  - $qMin = narrow\_range ? 1 : 0$
  - $qMax = 2^{num\_bits} - 1$
  - $scale = (max - min) / (qMax - qMin)$
  - $invScale = (qMax - qMin) / (max - min)$ (computed by an independent division, not reusing $1/scale$)
  - $nudgedZeroPoint = round(qMin - min / scale)$, clamped to $[qMin, qMax]$
  - $nudgedMin = (qMin - nudgedZeroPoint) \times scale$
  - $nudgedMax = (qMax - nudgedZeroPoint) \times scale$

  **Forward computation (kernel side):**

  $$
  clamped = clamp(x, nudgedMin, nudgedMax)
  $$

  $$
  y = floor(clamped \times invScale + 0.5) \times scale
  $$

  - When $min == max$, the output is all zeros (degenerate processing).

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>Input</td>
<td>Denotes the input tensor of the operator, corresponding to x in the formula; the rank range is [1, 8]; the shape is consistent with the output y.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>min</td>
<td>Input</td>
<td>Denotes the minimum value of the quantization range (scalar tensor), corresponding to min in the formula; the shape is (1,); it participates in gradient updates as a trainable variable.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>max</td>
<td>Input</td>
<td>Denotes the maximum value of the quantization range (scalar tensor), corresponding to max in the formula; the shape is (1,); it participates in gradient updates as a trainable variable.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>num_bits</td>
<td>Optional attribute</td>
<td>Denotes the quantization bit width, corresponding to num_bits in the formula; the value range is [2, 16]; the default value is 8.</td>
<td>INT</td>
<td>-</td>
</tr>
<tr>
<td>narrow_range</td>
<td>Optional attribute</td>
<td>Denotes whether to use the narrow quantization range; qMin=1 when true, and qMin=0 when false; the default value is false.</td>
<td>BOOL</td>
<td>-</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Denotes the computation output of fake quantization, corresponding to y in the formula; the shape is consistent with the input x.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- The data types of the input `x` and the output `y` support only FLOAT32, and their shapes must be exactly the same.
- The data types of the inputs `min` and `max` are FLOAT32, and their shapes must be (1,); when the shape of `min`/`max` is not (1,) (including empty tensors, e.g. with the shape (0,)), the operator validates it and rejects execution at the tiling stage (returning an error). An empty dimension of `x` is a valid input, and an empty input produces an empty output.
- The rank range of `x` is [1, 8].
- The value range of `num_bits` is [2, 16].
- When `min == max`, the output is all zeros (degenerate processing).
- Exception scenarios: for scenarios that violate the input constraints (such as a data type outside the supported list, an invalid or empty shape of `min`/`max`, a value of `num_bits` outside [2, 16], an unsupported data format, or a missing required input), the shape constraints of `min`/`max` are enforced by the operator at the tiling stage, which rejects non-conforming inputs; for other violations, the disposition (rejection of execution, an error report, or undefined computation results) and the specific error code are determined by the execution environment/framework, and no exact behavior or error code is guaranteed.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | - | Invokes the FakeQuantWithMinMaxVars operator by constructing the graph through the GE IR ([operator IR definition](op_graph/fake_quant_with_min_max_vars_proto.h)). `min`/`max` are scalar tensor inputs (trainable variables), and `num_bits`/`narrow_range` are attributes. The TensorFlow plugin directly maps the FakeQuantWithMinMaxVars op of TF to this operator. |
