# AdaLayerNormQuant

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: The AdaLayerNormQuant operator fuses AdaLayerNorm and the downstream quantization (currently only DynamicQuant is supported). This operator is mainly used to perform the quantization operation of adaptive layer normalization, that is, it normalizes the input data and quantizes it into low-precision integers to improve computation efficiency and reduce memory usage.

- Formula:

  1. First, perform LayerNorm normalization on input x:

     $$
     LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weight + bias
     $$

  2. Then adjust the normalization result through the adaptive parameters scale and shift:

     $$
     y = LayerNorm(x) * (1 + scale) + shift
     $$

  3. If smooth_scales is not empty:

     $$
     y = y \cdot smooth_scales
     $$

  4. Then compute the maximum absolute value of y and divide it by $(FP8\_MAX / HIF8\_MAX / INT8\_MAX)$ to compute the quantization factor for quantization into the FLOAT8/HIF8/INT8 formats:

     $$
     quant_scale = row\_max(abs(y)) / (FP8\_MAX / HIF8\_MAX / INT8\_MAX)
     $$

  5. Finally, divide y by the quantization factor and round to obtain the quantized output:

     $$
     out = round(y / quant_scale)
     $$

  Here, E(x) denotes the mean of the input, Var(x) denotes the variance of the input, and row_max represents taking the maximum value of each row.

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
      <td>Represents the input data to be processed, corresponding to `x` in the formula. The shape is [B..., S, H], where B supports 0 to 6 dimensions.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Represents the adaptive scaling parameter, corresponding to `scale` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Input</td>
      <td>Represents the adaptive offset parameter, corresponding to `shift` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Represents the normalization scaling parameter, corresponding to `weight` in the formula. The shape is [H], and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Represents the normalization offset parameter, corresponding to `bias` in the formula. The shape is [H], and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>Optional input</td>
      <td>Represents the smoothing weights of quantization, corresponding to `smooth_scales` in the formula. The shape is [H], and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the value added to the denominator to ensure numerical stability and to prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>A small positive number is recommended. The default value is 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the quantized output tensor, corresponding to `out` in the formula. The shape is consistent with the shape of input parameter `x`.</td>
      <td>INT8, FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_scale</td>
      <td>Output</td>
      <td>Represents the quantization factor, corresponding to `quant_scale` in the formula. The shape is [B..., S], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `x`, and S is consistent with the S dimension in `x`.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The data type of output `out` supports only INT8.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_ada_layer_norm_quant](examples/test_aclnn_ada_layer_norm_quant.cpp) | Calls the AdaLayerNormQuant operator through the [aclnnAdaLayerNormQuant](docs/aclnnAdaLayerNormQuant_en.md) interface. |
