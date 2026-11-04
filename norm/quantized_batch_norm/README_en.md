# QuantizedBatchNorm

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: it performs a dequantization computation on the input tensor, then performs normalization according to the input weight, bias, and epsilon, and finally performs quantization according to the input outputScale and outputZeroPoint.
- Formula:

  1. Dequantization computation:

     $$
     x' = (x - inputZeroPoint) * inputScale
     $$

  2. Normalization computation:

     $$
     y =\frac{x' - mean}{\sqrt{var + ϵ}} * γ + β
     $$

  3. Quantization computation:

     $$
     out = round(\frac{y}{outputScale} + outputZeroPoint)
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
      <td>The quantized data of the model input, corresponding to `x` in the formula.</td>
      <td>INT8, UINT8, INT32</td>
      <td>NCHW</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>The mean of the model input data, corresponding to `mean` in the formula. The shape is limited to 1D, and its length must match C in the input NCHW.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Input</td>
      <td>The variance of the model input data, corresponding to `var` in the formula. The shape is limited to 1D, and its length must match C in the input NCHW.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>Input</td>
      <td>An input scalar, the scaling coefficient of the model input data, corresponding to `inputScale` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_zero_point</td>
      <td>Input</td>
      <td>An input scalar, the offset of the model input data, corresponding to `inputZeroPoint` in the formula. The passed-in value must not exceed the upper and lower bounds of the data type of input, for example the bounds of INT8 are [-128,127].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_scale</td>
      <td>Input</td>
      <td>An input scalar, the scaling coefficient of the model output data, corresponding to `outputScale` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_zero_point</td>
      <td>Input</td>
      <td>An input scalar, the offset of the model output data, corresponding to `outputZeroPoint` in the formula. The passed-in value must not exceed the upper and lower bounds of the data type of input, for example the bounds of INT8 are [-128,127].</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>The weight for batch normalization, corresponding to `γ` in the formula. The shape is limited to 1D, and its length must match C in the NCHW data format of the input parameter input.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>The bias value for batch normalization, corresponding to `β` in the formula. The shape is limited to 1D, and its length must match C in the input NCHW.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>A small value added to the variance to avoid division by zero, corresponding to `ε` in the formula.</li><li>Default value: empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The quantized data of the model output, corresponding to `out` in the formula. Its data type, data format, and shape match the input `x`.</td>
      <td>INT8, UINT8, INT32</td>
      <td>NCHW</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_quantized_batch_norm](examples/test_aclnn_quantized_batch_norm.cpp) | Calls the QuantizedBatchNorm operator through the [aclnnQuantizedBatchNorm](docs/aclnnQuantizedBatchNorm_en.md) API. |
