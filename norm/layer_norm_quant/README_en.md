# LayerNormQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √    |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: The LayerNorm operator is a normalization operation commonly used by large models. The LayerNormQuant operator fuses the LayerNorm normalization output with the downstream quantization operator to reduce the move-in and move-out operations.
- Formula:
  - LayerNorm operation:

    $$
    y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
    $$

    $$
    E(x) = {\frac{1}{n} \sum_{i=1}^{n} x_i }
    $$

    $$
    Var(x) = {\frac{1}{n} \sum_{i=1}^{n} (x_i-E(x))^2 }
    $$

  - When quantMode is 0, the quantization mode is static quantization, and the output scaleOut has no actual meaning:

    $$
    res = y / scale + zeroPointsOptional
    $$

  - When quantMode is 1, the quantization mode is dynamic quantization:

    $$
    tmp = y * scale
    $$

    $$
    scaleOut = row\_max(abs(tmp))/dtypeMax
    $$

    $$
    res = round(y / scaleOut )
    $$

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
      <td>Denotes the x parameter in layer normalization, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the gamma parameter in layer normalization, corresponding to `gamma` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Corresponds to beta in the LayerNorm calculation formula and denotes the beta parameter in layer normalization, corresponding to `beta` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Denotes the scale input in the fused quantization computation, corresponding to `scale` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zeroPointsOptional</td>
      <td>Optional input</td>
      <td>Denotes the zeroPointsOptional input in the fused quantization computation. It is valid only when quantMode is 0, corresponding to `zeroPointsOptional` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quantMode</td>
      <td>Optional attribute</td>
      <td><ul><li>Quantization mode, used to determine whether the fused operator is a static or dynamic quantization operator during fusion. It corresponds to `quantMode` in the formula. The value is 0 (static quantization) or 1 (dynamic quantization).</li><li>The default value is 0.</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the epsilon in the corresponding LayerNorm, the value added to the denominator to ensure numerical stability. It corresponds to `epsilon` in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>res</td>
      <td>Output</td>
      <td>Denotes the quantized result of the LayerNorm result output y, corresponding to `res` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>Output</td>
      <td>Denotes the scaleOut result output of the dynamic quantization computation, corresponding to `scaleOut` in the formula. It is valid only when quantMode is 1.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:

    The data type FLOAT32 is not supported for the input parameters `x`, `gamma`, `beta`, and `scale`.
  - <term>Atlas inference products</term>:
    - The data type of the input parameters `x`, `gamma`, `beta`, and `scale` supports only FLOAT16.
    - The tail axis length of the input parameters `x`, `gamma`, and `beta` must be greater than or equal to 32 Bytes.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_layer_norm_quant](examples/test_aclnn_layer_norm_quant.cpp) | Calls the LayerNormQuant operator through the [aclnnLayerNormQuant](docs/aclnnLayerNormQuant_en.md) interface. |
