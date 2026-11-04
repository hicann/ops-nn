# AddRmsNormDynamicQuantV2

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean-subtraction part. The DynamicQuant operator performs symmetric dynamic quantization on input tensors. The AddRmsNormDynamicQuantV2 operator fuses Add, RmsNorm, and one or two DynamicQuant passes, and supports the optional bias beta and the output mask output_mask. Compared with AddRmsNormDynamicQuant, it additionally outputs the normalized result y3 (FLOAT32) and y4 (of the same type as the input) before adding beta and multiplying smooth_scale; this differs from the V1/V2 version distinction of the ACLNN interface.
- Formula:

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  yFP32=\begin{cases}
  y & outputMask[2]=True\ ||\ outputMask\ = null \\
  Invalid\ output & outputMask[2]=False
  \end{cases}
  $$

  $$
  y\_input=y+beta
  $$

  $$
  input1 =\begin{cases}
    y\_input \cdot smoothScale1Optional & \ \ smoothScale1Optional\ != null \\
    y\_input & \ \  smoothScale1Optional\ = null
    \end{cases}
  $$

  $$
  input2 =\begin{cases}
    y\_input \cdot smoothScale2Optional & \ \ smoothScale2Optional\ != null  \\
    y\_input & \ \ smoothScale2Optional\ = null
    \end{cases}
  $$

  $$
  scale1Out=\begin{cases}
    row\_max(abs(input1))/quantMax & outputMask[0]=True\ ||\ outputMask\ = null \\
    Invalid\ output & outputMask[0]=False
    \end{cases}
  $$

  $$
  y1Out=\begin{cases}
    quantize(input1/scale1Out) & outputMask[0]=True\ ||\ outputMask\ = null \\
    Invalid\ output & outputMask[0]=False
    \end{cases}
  $$

  $$
  scale2Out=\begin{cases}
    row\_max(abs(input2))/quantMax & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null) \\
    Invalid\ output & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ (smoothScale1Optional\ = null\ ||\ smoothScale2Optional\ = null))
    \end{cases}
  $$

  $$
  y2Out=\begin{cases}
    quantize(input2/scale2Out) & outputMask[1]=True\ ||\ (outputMask\ = null\ \&\ smoothScale1Optional\ != null\ \&\ smoothScale2Optional\ != null)\\
    Invalid\ output & outputMask[1]=False\ ||\ (outputMask\ = null\ \&\ (smoothScale1Optional\ = null\ ||\ smoothScale2Optional\ = null))
    \end{cases}
  $$

  Here, row\_max represents taking the maximum value of each row. y3 is the FLOAT32 value of the normalization result, and y4 is the value of this result converted to the input type; outputMask[2] and outputMask[3] control y3 and y4 respectively.

  `quantMax` is determined by the output type: INT8 is 127, INT4 is 7, FLOAT8_E4M3FN is 448, FLOAT8_E5M2 is 57344, and HIFLOAT8 is 32768. `quantize` performs round-to-nearest-even and saturated conversion for INT8/INT4, and performs the corresponding type conversion for floating-point quantization types without integer rounding. For an all-zero row, the scale is 0 and the quantized value is 0. When beta is absent, it is treated as 0; an invalid output is only a placeholder, and its value has no practical meaning. In graph mode, the INT4 output keeps the logical shape; the INT32 packed storage view applies only to the ACLNN interface and is not within the support scope of `dst_type`.

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
      <td><ul><li>Supports empty tensors.</li><li>Represents the source data tensor in the normalization process, corresponding to `x1` in the formula.</li><li>When the type of output `y1` or `y2` is INT4, the tail axis of `x1` must be divisible by 2.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the source data tensor in the normalization process, corresponding to `x2` in the formula. The shape and data type are the same as those of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the weight tensor in the normalization process, corresponding to `gamma` in the formula. The data type is the same as that of `x1`, and the shape must be consistent with the last dimension of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale1</td>
      <td>Optional input</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the smoothScale tensor used to obtain y1 in the quantization process, corresponding to `smoothScale1Optional` in the formula.</li><li>The shape and data type must be consistent with those of `gamma`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale2</td>
      <td>Optional input</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the smoothScale tensor used to obtain y2 in the quantization process, corresponding to `smoothScale2Optional` in the formula.</li><li>The shape and data type must be consistent with those of `gamma`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the bias term in the normalization process, corresponding to `beta` in the formula.</li><li>The shape and data type must be consistent with those of `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_mask</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the output mask, corresponding to `outputMask` in the formula. Only arrays of length 0 or length 4 are supported.</li><li>For specific constraints, see Constraints.</li><li>The default value is {}.</li></ul></td>
      <td>LISTBOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the output data types of `y1` and `y2`.</li><li>The input range is {2, 29, 34, 35, 36}, corresponding to {INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN} respectively.</li><li>The default value is 2.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the quantization output tensor, corresponding to `y1Out` in the formula.</li><li>If `y1` is a valid output, its shape is the same as that of input `x1`, and the data type is specified by `dst_type`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the quantization output tensor, corresponding to `y2Out` in the formula.</li><li>If `y2` is a valid output, its shape is the same as that of input `x1`, and the data type is specified by `dst_type`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y3</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the FLOAT32 output tensor of rmsNorm, corresponding to `yFP32` in the formula.</li><li>If `y3` is a valid output, its shape must be consistent with that of input `x1`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y4</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the output tensor of rmsNorm in the original input type, corresponding to `y` in the formula.</li><li>If `y4` is a valid output, its shape and data type must be consistent with those of input `x1`.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Represents the sum of x1 and x2, corresponding to `x` in the formula.</li><li>The shape and data type must be consistent with those of input `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Output of the first quantization pass, corresponding to `scale1Out` in the formula.</li><li>If this output is valid, its shape must be consistent with the shape of `x1` excluding the last dimension.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>Output</td>
      <td><ul><li>Supports empty tensors.</li><li>Output of the second quantization pass, corresponding to `scale2Out` in the formula.</li><li>If this output is valid, its shape must be consistent with the shape of `x1` excluding the last dimension.</li><li>For specific constraints, see Constraints.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A2 products</term>, <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>:
  - The data types of x1, x2, gamma, smooth_scale1, smooth_scale2, y4, and x do not support BFLOAT16.
  - The data types of y1 and y2 support only INT8.
  - The configurations of beta, output_mask, and dst_type do not take effect.
  - Whether y1 and y2 are output depends only on whether smooth_scale1 and smooth_scale2 are provided as inputs, and only y2 can be omitted.

## Constraints

- When output_mask is not empty:
  - If parameter smooth_scale1 has a value, output_mask[0] must be True. If parameter smooth_scale2 has a value, output_mask[1] must be True.
  - output_mask[0] and output_mask[1] cannot both be false.
  - The validity of each output is uniformly controlled by output_mask. When the position in output_mask is True, y and scale are valid outputs; when the position is False, y and scale are invalid outputs.

- When output_mask is empty:
  - If parameter smooth_scale2 has a value, parameter smooth_scale1 cannot be empty.
  - y1, y3, y4, and scale1 are always valid outputs.
  - y2 and scale2 are valid outputs only when both smooth_scale1 and smooth_scale2 are valid; otherwise, they are invalid outputs.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_add_rms_norm_dynamic_quant_v2](examples/test_geir_add_rms_norm_dynamic_quant_v2.cpp)  | Calls the AddRmsNormDynamicQuantV2 operator by building a graph through the [operator IR](op_graph/add_rms_norm_dynamic_quant_v2_proto.h).         |
