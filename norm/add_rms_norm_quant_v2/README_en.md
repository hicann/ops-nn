# AddRmsNormQuantV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | × |

## Function

- Operator function: RmsNorm is a normalization operation commonly used in large models. Compared with LayerNorm, RmsNorm removes the mean-subtraction part. The AddRmsNormQuant operator fuses the Add operator before RmsNorm and the output of RmsNorm normalization into 1 or 2 Quantize operators, reducing move-in and move-out operations. Compared with AddRmsNormQuant, the AddRmsNormQuantV2 operator adds a bias term in the RmsNorm computation, that is, `bias` in the formula, and adds an optional output `resOut`, which represents the RmsNorm normalization result before quantization (excluding bias).

- Formula:

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i + bias, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  resOut_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i
  $$

  - When div_mode is True:

    $$
    y1=round((y/scales1)+zero\_points1)
    $$

    $$
    y2=round((y/scales2)+zero\_points2)
    $$

  - When div_mode is False:

    $$
    y1=round((y*scales1)+zero\_points1)
    $$

    $$
    y2=round((y*scales2)+zero\_points2)
    $$

`round` indicates rounding and saturation processing based on the output data type.

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
      <td>Source data tensor in the normalization process, corresponding to `x1` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process, corresponding to `x2` in the formula. The shape must be the same as that of `x1`.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process, corresponding to `gamma` in the formula. The shape must be consistent with the dimensions of `x1` to be normalized.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>scales tensor for obtaining y1 in the quantization process, corresponding to `scales1` in the formula. When the value of parameter `div_mode` is True, the value of this parameter cannot be 0.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>scales tensor of the second quantization path, corresponding to `scales2` in the formula. This parameter is optional. When the value of parameter `div_mode` is True, the value of this parameter cannot be 0. The shape must be the same as that of `scales1`. For the default behavior, see <a href="#product-differences">Product Differences</a>.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>offset tensor of the first quantization path, corresponding to `zero_points1` in the formula. This parameter is optional. If it is not passed, the offset is not added. The shape must be the same as that of `gamma`.</td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>offset tensor of the second quantization path, corresponding to `zero_points2` in the formula. This parameter is optional. If it is not passed, the offset is not added. The shape must be the same as that of `gamma`. For valid combinations with `scales2`, see <a href="#product-differences">Product Differences</a>.</td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>bias term in the normalization process, corresponding to `bias` in the formula. This parameter is optional. If it is not passed, the bias is not added. The shape must be the same as that of `gamma`.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y1` in the formula. The shape must be the same as that of input `x1`/`x2`.</td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Output tensor of the second quantization path, corresponding to `y2` in the formula. The shape must be the same as that of input `x1`/`x2`. For the validity conditions of this output, see <a href="#product-differences">Product Differences</a>.</td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Optional output</td>
      <td>Sum of `x1` and `x2`, corresponding to `x` in the formula. The shape is the same as that of input `x1`/`x2`.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>resOut</td>
      <td>Optional output</td>
      <td>RmsNorm result excluding `bias`, corresponding to `resOut` in the formula. This output is generated when `output_res` is True. The shape is the same as that of input `x1`/`x2`.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>element-wise dimension to be quantized. Other dimensions are broadcast. Currently, only -1 is supported. Other values do not take effect.</li><li>Default value: -1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Used to prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>Default value: 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether the quantization computation uses division or multiplication, corresponding to `div_mode` in the formula.</li><li>Default value: True.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates the data types of `y1` and `y2`.</li><li>Default value: DT_INT8.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_res</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to output the RmsNorm result `resOut` excluding `bias`.</li><li>Default value: False.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

### Product Differences

The data types and data formats in the parameter table are the union of capabilities of all supported products. In the following table, "x-type parameters" include `x1`, `x2`, `gamma`, `bias`, `x`, and `resOut`; "scale parameters" include `scales1` and non-empty `scales2`; "zero point parameters" include non-empty `zero_points1` and `zero_points2`. Within the same valid combination, parameters of the same type have the same data type.

#### Data Type and Attribute Differences

| Product | Valid combinations of x-type parameters/scale parameters/zero point parameters | Data types of `y1` and `y2` (`dst_type`) | `div_mode` |
| --- | --- | --- | --- |
| <term>Ascend 950PR&950DT products</term> | FLOAT16/FLOAT32/INT32, BFLOAT16/FLOAT32/INT32, FLOAT32/FLOAT32/FLOAT32, FLOAT16/FLOAT16/FLOAT16, BFLOAT16/BFLOAT16/BFLOAT16, FLOAT16/FLOAT32/FLOAT32, BFLOAT16/FLOAT32/FLOAT32 | INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN; `y1` and `y2` have the same data type | True, False |
| <term>Atlas A3 products</term>, <term>Atlas A2 products</term> | FLOAT16/FLOAT32/INT32, BFLOAT16/BFLOAT16/BFLOAT16, BFLOAT16/FLOAT32/INT32 | INT8 | The native V2 path supports only True; False is supported when the aclnn API meets the V1 fallback conditions |
| <term>Atlas inference products</term> | FLOAT16/FLOAT32/INT32 | INT8 | The native V2 path supports only True; False is supported when the aclnn API meets the V1 fallback conditions |

For <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, and <term>Atlas inference products</term>, when the operator is invoked through the aclnn API and `div_mode` is False, the API does not enter the native V2 path. When output `x` is required, `resOut` is not output, and other parameters meet the V1 fallback path constraints, the API falls back to the V1 path and executes in multiplication mode. This fallback mechanism does not apply to GE graph mode; `div_mode` supports only True in GE graph mode.

#### Shape and Parameter Combination Differences

| Product | Static shape capability | Dynamic shape capability | Second quantization path | Other user-observable restrictions |
| --- | --- | --- | --- | --- |
| <term>Ascend 950PR&950DT products</term> | Both inputs and outputs support the ND format | Both inputs and outputs support the ND format | When either `scales2` or `zero_points2` is not empty, `y2` is a valid output. When only `zero_points2` is not empty, `scales2` is processed as 1, and $y2=round(y+zero\_points2)$ | - |
| <term>Atlas A3 products</term>, <term>Atlas A2 products</term> | Both inputs and outputs support the ND format | Both inputs and outputs support the ND format | `y2` is valid only when `scales2` is not empty; when `zero_points2` is not empty, `scales2` must not be empty at the same time | When the input is inf, the output is inf; when the input is NaN, the output is NaN |
| <term>Atlas inference products</term> | Both inputs and outputs support the ND format | Both inputs and outputs support the ND format | `y2` is valid only when `scales2` is not empty; when `zero_points2` is not empty, `scales2` must not be empty at the same time | inf and NaN are not supported for the input; the number of normalization elements contained in `gamma` cannot be less than 32 |

## Constraints

- The ACLNN API of <term>Ascend 950PR&950DT products</term> requires the rank of a non-null pointer tensor to be 1–8, and the actual dimension length must not exceed INT32_MAX; in graph mode, the rank must not exceed 8. `gamma` corresponds to the complete normalization tail axis. The compatibility handling of the historical `[1,N]` gamma/beta alias belongs to the ACLNN API and does not change the tail-axis reduction rule in graph mode.
- The ACLNN API of <term>Ascend 950PR&950DT products</term> requires `xOut` to be a non-null pointer; `rmsNormOut` is an optional output and is independent of `betaOptional`. In graph mode, `output_res` determines whether to output `resOut`; `resOut` is the normalization result before bias addition and quantization. The second quantization path is enabled when either `scales2` or `zero_points2` exists; when both are absent, `y2` in graph mode is an invalid placeholder output with shape `[1]`, and `y2Out` of the ACLNN API does not write a valid quantization result and should not participate in numerical comparison.
- In the GE graph definition, an unknown dimension is represented by `-1` and an unknown rank is represented by `[-2]`; they must be specialized before execution and meet the shape constraints of the corresponding product. The ACLNN API accepts only concrete tensor descriptions. In graph mode of <term>Ascend 950PR&950DT products</term>, tensors are processed as contiguous ND storage; the ACLNN API supports non-contiguous input and output views in the parameter table, makes the input contiguous, and writes back the enabled outputs.
- <term>Ascend 950PR&950DT products</term> support empty tensors with valid shapes. No element computation is performed, and the output retains the shape inference result.

The following four output and quantization parameter combination restrictions apply to <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Atlas inference products</term>:

- For the optional outputs `x` and `resOut`, you must select exactly one of them for output.
- When `y2` needs to be output, the shape of `gamma` must be the same as that of `scale1` and consistent with the dimensions of `x1` to be normalized, and the only optional output is `x`.
- When `x` needs to be output, and the shapes of `scales1` and `zero_points1` are [1], and the shape of `gamma` is 1-dimensional and equal to the last dimension of x1, or the shape of `gamma` is 2-dimensional with the first dimension being 1 and the second dimension being the last dimension of `x1`, `scales2` and `zero_points2` do not take effect.
- When `resOut` needs to be output, the shapes of `scales1` and `zero_points1` are [1], and the shape of `gamma` is 1-dimensional and equal to the last dimension of x1, or the shape of `gamma` is 2-dimensional with the first dimension being 1 and the second dimension being the last dimension of `x1`, and `bias` and `x` must be passed as null pointers. In this case, `scales2` and `zero_points2` do not take effect.

- **Dimension boundary description**

  For the shapes of parameters `x1`, `x2`, `gamma`, `bias`, `scales1`, `scales2`, `zero_points1`, `zero_points2`, `y1`, `y2`, `x`, and `resOut`, the size of each dimension must not exceed the INT32 maximum value 2147483647.

- **Data format description**

  For all input and output tensors, the ND format is recommended as the data format. Other data formats are converted to the ND format by the framework by default for processing.

- The RMS reduction of <term>Ascend 950PR&950DT products</term> covers the complete tail axis corresponding to `gamma`, and the number of reduced elements is the product of these dimensions; `axis=-1` is an attribute of the quantization API and does not narrow the RMS reduction of multi-dimensional `gamma` to the last dimension. When bias exists, its shape is the same as that of gamma.
- Kernel/GEIR parameter check failures are reported through the failure status and diagnostic information of graph construction or tiling; 161001 and 161002 of the ACLNN API are return codes at the API layer and are not used as unified return codes of Kernel/GEIR.

## Invocation Description

The preceding data type combinations apply to the Kernel/GE graph API. For the input types and combinations supported by the aclnn API, see [aclnnAddRmsNormQuantV2 API Description](docs/aclnnAddRmsNormQuantV2_en.md#constraints).

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_add_rms_norm_quant_v2](examples/test_aclnn_add_rms_norm_quant_v2.cpp) | Invoke the AddRmsNormQuantV2 operator through the two-phase API in [aclnnAddRmsNormQuantV2 API Description](docs/aclnnAddRmsNormQuantV2_en.md). |
| GE graph mode | - | Invoke the AddRmsNormQuantV2 operator through graph construction using the [AddRmsNormQuantV2 operator IR](op_graph/add_rms_norm_quant_v2_proto.h). |
