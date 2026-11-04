# SituMxQuant

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

- Operator function: fuses the Situ activation function and dynamic MX (Microscaling) quantization into one operator.

- Formula:

  1. Situ activation:

     $$
     situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
     $$

     When linear_beta > 0:

     $$
     up = linear\_beta \times \tanh(up / linear\_beta)
     $$

     $$
     situOut = situ_a \times up
     $$

     Here, when activate_left is true, gate takes the first half of x and up takes the second half; when activate_left is false, gate takes the second half of x and up takes the first half.

  2. MX quantization (OCP algorithm):

     $$
     shared\_exp = floor(log2(max(|situOut_i|))) - emax
     $$

     $$
     y\_scale = 2^{shared\_exp}  (E8M0)
     $$

     $$
     y = cast\_to\_fp8(situOut / y\_scale)
     $$

## Parameters

<table style="undefined;table-layout: fixed; width: 951px"><colgroup>
  <col style="width: 121px">
  <col style="width: 144px">
  <col style="width: 313px">
  <col style="width: 257px">
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
      <td>Data to be processed, corresponding to x in the formula. The last dimension must be a multiple of 2. The input does not support containing ±inf or nan.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td>beta parameter of the Situ activation, corresponding to β in the formula. It must be greater than 0. The default value is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td>linear_beta parameter of the Situ activation, corresponding to linear_beta in the formula. It is not enabled when the value is ≤ 0. The default value is 0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td>Indicates whether gate takes the first half or the second half of x, corresponding to activate_left in the formula. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Attribute</td>
      <td>Denotes the quantization axis, the axis for computing max and scale in the formula. Currently only -1 is supported. The default value is -1.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Attribute</td>
      <td>Denotes the data type of output y, corresponding to the target type of cast_to_fp8 in the formula: 36=FLOAT8_E4M3FN and 35=FLOAT8_E5M2. The default value is 36.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>Attribute</td>
      <td>Denotes the rounding mode of quantization, the rounding method of cast in the formula. "rint", "round", and "floor" are supported. FP8 outputs support only "rint". The default value is "rint".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Quantized output, corresponding to y in the formula.</td>
      <td>FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y_scale</td>
      <td>Output</td>
      <td>Scale of MX quantization (E8M0 format), corresponding to y_scale in the formula.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The last dimension of x must be a multiple of 2.
- x must have at least 1 dimension.
- axis currently supports only -1 (tail-axis quantization).
- dst_type supports 36 (FLOAT8_E4M3FN) or 35 (FLOAT8_E5M2).
- round_mode must be "rint".
- The data type of y must match dst_type, and the data type of y_scale must be FLOAT8_E8M0.
- The shapes of y and y_scale must be consistent with the derived results (see the Parameters section).
- The shape constraints of y_scale are as follows:
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - y_scale.shape = x.shape[:-1] + [scaleNum, 2]

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_situ_mx_quant](./examples/arch35/test_aclnn_situ_mx_quant.cpp) | Invokes the SituMxQuant operator through the [aclnnSituMxQuant](./docs/aclnnSituMxQuant_en.md) interface.    |
| Graph-mode invocation | - | Invokes the SituMxQuant operator through graph construction with the [operator IR](./op_graph/situ_mx_quant_proto.h). |
