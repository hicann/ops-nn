# AddLayerNormQuant

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
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: The LayerNorm operator is a normalization operation commonly used in large models. The AddLayerNormQuant operator fuses the Add operator before LayerNorm and the LayerNorm normalization output into one or two downstream quantization operators, reducing data move-in and move-out operations. The quantization operators downstream of LayerNorm can be the Quantize, AscendQuantV2, or DynamicQuant operator, and the specific quantization operator type is determined by the attr input parameters divMode and quantMode. When there are 2 quantization operators downstream, the operator types, input/output dtype combinations, and optional input combinations of the 2 quantization operators must be completely identical.
- Formula*:

  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$

  - When quantMode is "static", the outputs outScales1 and outScales2 have no practical meaning. Depending on the divMode input, the fused quantization operator may be Quantize or AscendQuantV2:
    - When divMode is true, the fused quantization operator is Quantize, and the formula is as follows:

      $$
      y1 = round(y / scales1 + zeroPoints1)
      $$

      $$
      y2 = round(y / scales2 + zeroPoints2), \quad \text{if and only if scales2 exists}
      $$

    - When divMode is false, the fused quantization operator is AscendQuantV2, and the formula is as follows:

      $$
      y1 = round(y * scales1 + zeroPoints1)
      $$

      $$
      y2 = round(y * scales2 + zeroPoints2), \quad \text{if and only if scales2 exists}
      $$

  - When quantMode is "dynamic", the inputs zeroPoints1 and zeroPoints2 have no practical meaning. The fused quantization operator is DynamicQuant, and divMode is invalid in this case:
    - If neither scales1 nor scales2 has an input, the y2 and scale2 outputs have no practical meaning and can be ignored. The formula is as follows:

      $$
      outScales1 = row\_max(abs(y))/127
      $$

      $$
      y1 = round(y / outScales1)
      $$

    - If only scales1 is input, the y2 and scale2 outputs have no practical meaning and can be ignored. The formula is as follows:

      $$
      tmp1 = y * scales1
      $$

      $$
      outScales1 = row\_max(abs(tmp1))/127
      $$

      $$
      y1 = round(y / outScales1)
      $$

    - If both scales1 and scales2 exist, the y2 and scale2 outputs are valid. The formula is as follows:

      $$
      tmp1 = y * scales1, \quad tmp2 = y * scales2
      $$

      $$
      outScales1 = row\_max(abs(tmp1))/127, \quad outScales2 = row\_max(abs(tmp2))/127
      $$

      $$
      y1 = round(y / outScales1),\quad y2 = round(y / outScales2)
      $$

    row\_max denotes the maximum value of each row.

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
      <td>x1</td>
      <td>Input</td>
      <td>Input of the addition in AddLayerNorm, corresponding to `x1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input of the addition in AddLayerNorm, corresponding to `x2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma parameter in layer normalization, corresponding to `gamma` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>beta parameter in layer normalization, corresponding to `beta` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Input of the addition in AddLayerNormQuant, corresponding to `bias` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Optional input</td>
      <td>scale/smooth input of the first fused quantization operator, corresponding to `scales1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>scale/smooth input of the second fused quantization operator, corresponding to `scales2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>zeroPoints input of the first fused quantization operator, corresponding to `zeroPoints1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>zeroPoints input of the second fused quantization operator, corresponding to `zeroPoints2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Determines whether the operator fuses a static or dynamic quantization operator, corresponding to `quantMode` in the formula. The value can be "static" or "dynamic".</li><li>Default value: "dynamic".</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>Default value: 1e-5.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to enable the output of x=x1+x2+bias.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether the static quantization method for processing scale is multiplication or division, corresponding to `divMode` in the formula.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y1` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y2` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales1</td>
      <td>Output</td>
      <td>Dynamic quantization scaling result computed with scales1, corresponding to `outScales1` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales2</td>
      <td>Output</td>
      <td>Dynamic quantization scaling result computed with scales2, corresponding to `outScales2` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | - | Invokes the AddLayerNormQuant operator through the [aclnnAddLayerNormQuant](docs/aclnnAddLayerNormQuant_en.md) interface. |
| Graph-mode invocation | [test_geir_add_layer_norm_quant](examples/test_geir_add_layer_norm_quant.cpp)  | Invokes the AddLayerNormQuant operator through graph construction with the [operator IR](op_graph/add_layer_norm_quant_proto.h).         |
