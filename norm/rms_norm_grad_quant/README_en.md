# RmsNormGradQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: RmsNormGrad computes the gradient of RmsNorm, that is, it computes the gradient of the input tensor during backpropagation. The RmsNormGradQuant operator fuses the two operators RmsNormGrad and Quantize. After RmsNormGrad computes dx, the quant computation is performed, which reduces the move-in and move-out operations.

- Formula:

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  - When div\_mode is True:

    $$
    dx_i\_quant=round((dx_i / scales\_x) + offset\_x)
    $$

  - When div\_mode is False:

    $$
    dx_i\_quant=round((dx_i * scales\_x) + offset\_x)
    $$

  $$
  dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
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
      <td>dy</td>
      <td>Input</td>
      <td><ul><li>Represents the gradient back-propagated, corresponding to <code>dy</code> in the formula.</li><li>The shape supports 2-8 dimensions.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Represents the input of the forward operator, that is, the data to be normalized, corresponding to <code>x</code> in the formula.</li><li>Its shape and dtype are the same as those of dy.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td><ul><li>Represents the intermediate computation result of the forward operator, corresponding to the reciprocal of <code>Rms(x)</code> in the formula.</li><li>The shape must satisfy rstd_shape = x_shape[0:n], n &lt; x_shape.dims(), and n is the same as the n of gamma.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td><ul><li>Represents the scale factor (weight) used by the forward operator for normalization, corresponding to <code>g</code> in the formula.</li><li>The shape must satisfy gamma_shape = x_shape[n:], n &lt; x_shape.dims().</li><li>Its dtype is the same as dy or is FLOAT32.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales_x</td>
      <td>Input</td>
      <td><ul><li>Represents the quantization scale factor of the input gradient, corresponding to <code>scales_x</code> in the formula.</li><li>The shape is [1] with 1 dimension.</li><li>Its dtype is the same as dy or is FLOAT32.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset_x</td>
      <td>Optional input</td>
      <td><ul><li>Represents the quantization zero point of the input gradient, corresponding to <code>offset_x</code> in the formula.</li><li>The shape is [1] with 1 dimension.</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Attribute</td>
      <td><ul><li>Quantization mode.</li><li>Only "static" is supported, which indicates the static quantization mode.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Attribute</td>
      <td><ul><li>Parameter in the formula that determines whether the quantization formula uses division, corresponding to <code>div_mode</code> in the formula.</li><li>True and False are supported.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the type of dx after data conversion.</li><li>The input range is {2, 34}, corresponding to {INT8, HIFLOAT8} respectively.</li><li>The default value is 2.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td><ul><li>Represents the quantized gradient of input <code>x</code>, corresponding to <code>dx_quant</code> in the formula.</li><li>Its shape is the same as the shape of the input parameter <code>dy</code>.</li></ul></td>
      <td>INT8, HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>Output</td>
      <td><ul><li>Represents the gradient of <code>gamma</code>, corresponding to <code>dg</code> in the formula.</li><li>Its shape is the same as the shape of the input parameter <code>gamma</code>.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- <term>Ascend 950PR&950DT products</term>:
  - All input tensors support empty tensors.
  - dy, x, rstd, and gamma support non-contiguous tensors; scales\_x and offset\_x do not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_rms_norm_grad_quant](examples/arch35/test_aclnn_rms_norm_grad_quant.cpp) | Invokes the RmsNormGradQuant operator through the [aclnnRmsNormGradQuant](docs/aclnnRmsNormGradQuant_en.md) interface. |
| Graph-mode invocation | -  | Invokes the RmsNormGradQuant operator through the operator IR-based [graph construction](op_graph/rms_norm_grad_quant_proto.h) method.         |
