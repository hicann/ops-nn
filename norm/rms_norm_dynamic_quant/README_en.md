# RmsNormDynamicQuant

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
|  <term>Kirin X90 processor series products</term>       |     ×    |
|  <term>Kirin 9030 processor series products</term> | × |

## Function

- Operator function: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the part that subtracts the mean. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The RmsNormDynamicQuant operator fuses RmsNorm normalization and DynamicQuant dynamic quantization, which reduces the move-in and move-out operations.

- Formula:

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - If smooth_scales is not input, quantization is performed directly on the RmsNorm output:

  $$
   scaleOut=row\_max(abs(y))/max\_val
  $$

  $$
   yOut=round(y/scaleOut)
  $$

  - If smooth_scales is input, smooth scaling is performed first and then quantization:

  $$
    input = y\cdot smooth\_scales
  $$

  $$
   scaleOut=row\_max(abs(input))/max\_val
  $$

  $$
   yOut=round(input/scaleOut)
  $$

  Where row\_max means computing the maximum value of each row. max\_val is 127 for INT8.

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
      <td>Source data tensor in the normalization process. It is x in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process, gamma in the formula. Its shape must be the same as the last dimension of x.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scales</td>
      <td>Optional input</td>
      <td>Represents the smoothScale tensor used in the quantization process, smooth_scales in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Bias term in the normalization process, beta in the formula. Its shape and dtype must be the same as those of gamma.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division-by-zero errors. It is epsilon in the formula and must be greater than zero.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the data type enumeration value of output y.</li><li>The default value is DT_INT8.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the quantization output tensor, yOut in the formula. Its shape is the same as x.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Output</td>
      <td>Quantization scale output, scaleOut in the formula. Its shape is the shape of x after the last dimension is removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Input shape limit: the last dimension of input `x` must be less than or equal to 8192; otherwise accuracy problems may occur.

- Input value range limit: `x` does not support all-zero inputs.

## Invocation Description

| Invocation method   | Invocation sample           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_rms_norm_dynamic_quant](examples/test_aclnn_rms_norm_dynamic_quant.cpp) | Invokes the RmsNormDynamicQuant operator through the [aclnnRmsNormDynamicQuant](docs/aclnnRmsNormDynamicQuant_en.md) interface |
| GE graph mode | [test_geir_rms_norm_dynamic_quant](examples/test_geir_rms_norm_dynamic_quant.cpp) | Invokes the RmsNormDynamicQuant operator through the operator IR-based [graph construction](op_graph/rms_norm_dynamic_quant_proto.h) method |
| PyTorch API | - | Invokes the RmsNormDynamicQuant operator through the [cann_ops_nn.rms_norm_dynamic_quant](docs/torchapi_rms_norm_dynamic_quant_en.md) interface |
