# RmsNormQuantV3

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

- Operator function: the RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction part. The RmsNormQuantV3 operator fuses the RmsNorm operator and the output of RmsNorm normalization into one or two Quantize operators, reducing the data transfer in and out operations. In addition, it adds the Rstd output on the basis of the RmsNormQuantV2 operator.

- Formula:

  $$
  y_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - When divMode is True:

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$

  - When divMode is False:

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
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
      <td>Denotes the source data tensor in the normalization process, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the weight tensor in the normalization process, corresponding to `g` in the formula. The shape must be consistent with the dimensions of x to be normed (layer normalization).</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>Denotes the scales tensor used to obtain y1 in the quantization process, corresponding to `scales1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>Denotes the scales tensor used to obtain y2 in the quantization process, corresponding to `scales2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>Denotes the offset tensor used to obtain y1 in the quantization process, corresponding to `zero_points1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>Denotes the offset tensor used to obtain y2 in the quantization process, corresponding to `zero_points2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Denotes the bias term in the normalization process, `beta` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>The parameter in the formula that determines whether the quantization formula uses division, corresponding to `divMode` in the formula.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the type of y after the specified data conversion, corresponding to the data type of `y1` or `y2` in the formula.</li><li>The input range is {2, 29, 34, 35, 36}, corresponding to {INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN} respectively.</li><li>The default value is 2.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_rstd</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to output a valid rstd.</li><li>Supports True and False.</li><li>The default value is False.</li><li>When output_rstd is False, the operator does not write out rstd, and rstd is an invalid placeholder output.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Denotes the quantized output tensor, corresponding to `y1` in the formula.</td>
      <td>INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Denotes the quantized output tensor, corresponding to `y2` in the formula. When `scales2` is not passed in, this output is an invalid placeholder tensor and does not participate in the computation; the placeholder shape is [0] when `x` is empty, and [1] otherwise.</td>
      <td>INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td><ul><li>Denotes the reciprocal of the standard deviation after normalization, corresponding to the reciprocal of Rms(x) in the formula.</li><li>When output_rstd is True, the shape is consistent with the non-reduced dimensions of x, and the size of each reduced dimension is 1; the number of reduced dimensions equals the number of dimensions of gamma.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- When the input shape and data type constraints are satisfied, `x` supports an empty tensor and all valid outputs are empty tensors, and the GE framework does not execute the Kernel; the disabled `y2` and `rstd` are also deduced as empty tensors. The scenario where `output_rstd` is True and `rstd` is non-empty after reduction is not supported.

- <term>Ascend 950PR&950DT products</term>:
  - When the data type of `x` is FLOAT32, the data types of `scales1`, `scales2`, `zero_points1`, and `zero_points2` can only be FLOAT32; when the data type of `x` is FLOAT16 or BFLOAT16 and the data types of `scales1` and `scales2` are FLOAT32, the data types of `zero_points1` and `zero_points2` can only be FLOAT32 or INT32; otherwise, the data types of `scales1` and `scales2` need to be consistent with the data type of `x`, both being FLOAT16 or BFLOAT16, and the data types of `zero_points1` and `zero_points2` also need to be consistent with the data type of `x` or be INT8.
  - When the data type of `y` is INT4, the last dimension of `x`, `gamma`, and `beta` must be an even number.
  - `zero_points2` can serve as an optional input only when `scales2` exists.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_rms_norm_quant_v3](../rms_norm_quant_v3/examples/test_aclnn_rms_norm_quant_v3.cpp) | Invokes the RmsNormQuantV3 operator through the [aclnnRmsNormQuantV3](../rms_norm_quant_v3/docs/aclnnRmsNormQuantV3_en.md) API. |
| Graph mode | -  | Invokes the RmsNormQuantV3 operator by constructing the graph through the [operator IR](op_graph/rms_norm_quant_v3_proto.h).         |
