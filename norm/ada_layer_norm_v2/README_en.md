# AdaLayerNormV2

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |    ×    |
|  <term>Atlas training products</term>    |    ×    |

## Function

- Operator function: The AdaLayerNormV2 operator fuses LayerNorm and the downstream Add and Mul, and adjusts the normalization process through the adaptive parameters scale and shift. Compared with the AdaLayerNorm operator, the output adds 2 parameters (the mean of the input and the reciprocal of the standard deviation of the input), and the supported data types of weight and bias add corresponding constraints.
- Formula:

  $$
  out = LayerNorm(x) * (1 + scale) + shift
  $$

  LayerNorm computation formula:

  $$
  mean = E(x)
  $$

  $$
  rstd = {1.0\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  LayerNorm(x) = (x-mean) * rstd * weight + bias
  $$

  E(x) denotes the mean of the input, and Var(x) denotes the variance of the input.

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
      <td>Input tensor of the computation, corresponding to `x` in the formula. The shape is [B..., S, H], where B supports 0 to 6 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Adaptive scaling parameter, corresponding to `scale` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number and sizes of the B dimensions are consistent with those of B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Input</td>
      <td>Adaptive offset parameter, corresponding to `shift` in the formula. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions. The number and sizes of the B dimensions are consistent with those of B in `x`, and H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>Normalization scaling parameter, corresponding to `weight` in the formula. The shape is [H], where H is consistent with the H dimension in `x`. Its data type is the same as the data type of the input parameter `x` or is FLOAT32. When `bias` exists, `weight` and `bias` have the same data type. The shape is [H], where H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Normalization offset parameter, corresponding to `bias` in the formula. Its data type is the same as the data type of the input parameter `x` or is FLOAT32. When `bias` exists, `weight` and `bias` have the same data type. The shape is [H], where H is consistent with the H dimension in `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>A small positive number is recommended. Default value: 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor of the computation, corresponding to `out` in the formula. Its shape is consistent with that of `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Mean after normalization, corresponding to `mean` in the formula. The shape is [B..., S, 1]. The last dimension is fixed to 1, and the other dimension sizes are consistent with those of `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Reciprocal of the standard deviation after normalization, corresponding to `rstd` in the formula. The shape is [B..., S, 1]. The last dimension is fixed to 1, and the other dimension sizes are consistent with those of `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_ada_layer_norm_v2](examples/test_aclnn_ada_layer_norm_v2.cpp) | Invokes the AdaLayerNormV2 operator through the [aclnnAdaLayerNormV2](docs/aclnnAdaLayerNormV2_en.md) interface. |
