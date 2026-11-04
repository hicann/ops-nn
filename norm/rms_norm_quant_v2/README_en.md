# RmsNormQuantV2

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

- Operator function: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the part of subtracting the mean. The RmsNormQuantV2 operator fuses the RmsNorm operator and the output of RmsNorm normalization into 1 or 2 Quantize operators, reducing the move-in and move-out operations.

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
      <td>Source data tensor in the normalization process, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process, corresponding to `g` in the formula. The shape must be consistent with the dimensions of x to be normalized (layer normalization).</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>Scales tensor for obtaining y1 in the quantization process, corresponding to `scales1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>Scales tensor for obtaining y2 in the quantization process, corresponding to `scales2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>Offset tensor for obtaining y1 in the quantization process, corresponding to `zero_points1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>Offset tensor for obtaining y2 in the quantization process, corresponding to `zero_points2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Bias term in the normalization process, corresponding to `beta` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division by 0 errors, corresponding to `epsilon` in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Parameter in the formula that determines whether the quantization formula uses division, corresponding to `divMode` in the formula.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the type of y after data conversion, corresponding to the data type of `y1` or `y2` in the formula.</li><li>The input range is {2, 29, 34, 35, 36}, corresponding to {INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN} respectively.</li><li>The default value is 2.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y1` in the formula.</td>
      <td>INT8, INT4, FLOAT8, HIF8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y2` in the formula. When `scales2` is not passed in, this output is an invalid placeholder tensor and does not participate in the computation. When `x` is empty, the placeholder shape is [0]; otherwise, it is [1].</td>
      <td>INT8, INT4, FLOAT8, HIF8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- When the input shape and data type constraints are met, `x` is supported as an empty tensor with all outputs as empty tensors, and the GE framework does not execute the kernel. The disabled `y2` is also deduced as an empty tensor.

- <term>Ascend 950PR&950DT products</term>:
  - When the data type of `x` is FLOAT32, the data types of `scales1`, `scales2`, `zero_points1`, and `zero_points2` can only be FLOAT32. When the data type of `x` is FLOAT16 or BFLOAT16 and the data types of `scales1` and `scales2` are FLOAT32, the data types of `zero_points1` and `zero_points2` can only be FLOAT32 or INT32. Otherwise, the data types of `scales1` and `scales2` must be consistent with the data type of `x` and be both FLOAT16 or BFLOAT16, and the data types of `zero_points1` and `zero_points2` must also be consistent with the data type of `x` or be INT8.
  - When the data type of `y` is INT4, the last dimensions of `x`, `gamma`, and `beta` must be even numbers.
  - `zero_points2` can be used as an optional input only when `scales2` exists.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_rms_norm_quant](../rms_norm_quant/examples/test_aclnn_rms_norm_quant.cpp) | Calls the RmsNormQuantV2 operator through the [aclnnRmsNormQuant](../rms_norm_quant/docs/aclnnRmsNormQuant_en.md) interface. |
| Graph mode | [test_geir_rms_norm_quant_v2](examples/test_geir_rms_norm_quant_v2.cpp) | Calls the RmsNormQuantV2 operator by building a graph through the [operator IR](op_graph/rms_norm_quant_v2_proto.h). |
