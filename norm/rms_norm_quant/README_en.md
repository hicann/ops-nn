# RmsNormQuant

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :---------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                                                |    √    |
| <term>Atlas A3 products</term>                        |    √    |
| <term>Atlas A2 products</term> |    √    |
| <term>Atlas 200I/500 A2 inference products</term>                                         |    √    |
| <term>Atlas inference products</term>                                                |    √    |
| <term>Atlas training products</term>                                                 |    ×    |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: The RmsNorm operator is a standardization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean-subtraction part. The RmsNormQuant operator fuses the RmsNorm operator and the Quantize operator after RmsNorm, reducing data move-in and move-out operations.
- Formula:

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i + b_i, \quad \text { where }   \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y=round((quant\_in*scale)+offset)
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
      <td>Source data tensor in the standardization process, which is `x` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling tensor in the standardization process, which is `g` in the formula. The shape supports 1 to 2 dimensions. Its data type must be consistent with that of `x`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Offset tensor in the standardization process, which is `b` in the formula. Its shape and data type must be consistent with those of `gamma`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>scale tensor used to obtain y in the quantization process, which is `scale` in the formula. The shape is 1 and the dimension is 1.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>offset tensor used to obtain y in the quantization process, which is `offset` in the formula. Its shape must be consistent with that of `scale`.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to `eps` in the formula.</li><li>Default value: 0.1f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>high_precision_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to use high-precision mode. This parameter is reserved and not used yet.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gemma_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to use gemma mode. This parameter is reserved and not used yet.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final quantization output tensor, corresponding to `y` in the formula. Its shape must be consistent with that of the input `x`.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>, <term>Atlas 200I/500 A2 inference products</term>, <term>Kirin X90 processor series products</term>, and <term>Kirin 9030 processor series products</term>: the data types of the input parameters `x`, `gamma`, `beta`, and `scale` do not support BFLOAT16.

## Constraints

- <term>Atlas inference products</term>: the trailing axis lengths of x and y, and the trailing axis length of gamma, must be greater than or equal to 32 Bytes.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_rms_norm_quant](examples/test_aclnn_rms_norm_quant.cpp) | Invokes the RmsNormQuant operator through the [aclnnRmsNormQuant](docs/aclnnRmsNormQuant_en.md) interface. |
