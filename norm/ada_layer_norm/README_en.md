# AdaLayerNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: The AdaLayerNorm operator fuses LayerNorm and the downstream Add and Mul, and adjusts the normalization process through the adaptive parameters scale and shift.

- Formula:

  $$
  out = LayerNorm(x) * (1 + scale) + shift
  $$

  LayerNorm formula:

  $$
  LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weight + bias
  $$

  Here, E(x) denotes the mean of the input, and Var(x) denotes the variance of the input.

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
      <td>Represents the input data to be normalized, corresponding to `x` in the formula. The shape is [B..., S, H], where B supports 0 to 6 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Represents the adaptive scaling parameter, corresponding to `scale` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Input</td>
      <td>Represents the adaptive offset parameter, corresponding to `shift` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Represents the normalization scaling parameter, corresponding to `weight` in the formula. The shape is [H], and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Represents the normalization offset parameter, corresponding to `bias` in the formula. The shape is [H], and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
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
      <td>Represents the normalized result, corresponding to `out` in the formula. The shape is consistent with `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_ada_layer_norm](examples/test_aclnn_ada_layer_norm.cpp) | Calls the AdaLayerNorm operator through the [aclnnAdaLayerNorm](docs/aclnnAdaLayerNorm_en.md) interface. |
