# AddRmsNormQuant

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: RmsNorm is a normalization operation commonly used in large models. Compared with LayerNorm, it removes the mean-subtraction part. The AddRmsNormQuant operator fuses the Add operator before RmsNorm and the normalized output of RmsNorm into one or two Quantize operators, reducing data move-in and move-out operations.

- Formula:

  $$
  x_i={x1}_i+{x2}_i
  $$

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

`round` denotes rounding and saturation processing according to the output data type.

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
      <td>Represents the source data tensor in the normalization process, corresponding to `x1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Represents the source data tensor in the normalization process, corresponding to `x2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Represents the weight tensor in the normalization process, corresponding to `g` in the formula. The shape must be consistent with the dimensions of x1 that require normalization (layer normalization).</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>Represents the scales tensor used to obtain y1 in the quantization process, corresponding to `scales1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>Represents the scales tensor used to obtain y2 in the quantization process, corresponding to `scales2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>Represents the offset tensor used to obtain y1 in the quantization process, corresponding to `zero_points1` in the formula.</td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>Represents the offset tensor used to obtain y2 in the quantization process, corresponding to `zero_points2` in the formula.</td>
      <td>INT32, FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Represents the bias term in the normalization process, `beta` in the formula. Optional; the shape is consistent with gamma and the dtype is consistent with x1. This input belongs to the Kernel/GEIR interface, and the V1 ACLNN interface does not provide beta.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the element-wise axis to be quantized. Other axes are broadcast. The specified axis cannot exceed the number of dimensions of input x. Currently, only -1 is supported, and passing other values does not take effect.</li><li>The default value is -1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
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
      <td><ul><li>Parameter in the formula that determines whether the quantization formula uses division, corresponding to `divMode` in the formula.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Represents the quantization output tensor, corresponding to `y1` in the formula.</td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Represents the quantization output tensor, corresponding to `y2` in the formula.</td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Represents the sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: The data types of x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, beta, and x do not support BFLOAT16, and the data types of output parameters y1 and y2 support only INT8.

  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data types of x1, x2, gamma, x, and beta do not support FLOAT32, and the data types of output parameters y1 and y2 support only INT8.

  - <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>:
    - The data types of x1, x2, gamma, beta, and x support only FLOAT16.
    - The data types of scales1 and scales2 support only FLOAT32.
    - The data types of zero_points1 and zero_points2 support only INT32.
    - The data types of y1 and y2 support only INT8.

## Constraints

- For the ACLNN interface, a non-null pointer tensor requires a rank of 1–8, and the size of each dimension must not be greater than INT32_MAX (2147483647); a rank-0 scalar is not within the support scope of this interface. In graph mode, the rank of inputs is not greater than 8, `gamma` corresponds to the complete normalization tail axis of `x1`, the shapes of `x1`, `x2`, `y1`, and `x` are the same, and the quantization parameters must satisfy the shape relationships in Parameters.
- Dynamic shape/rank belongs to the GE graph compilation capability: in a graph definition, an unknown dimension is represented by `-1`, and an unknown rank is represented by `[-2]`. Before execution, you must concretize them into an actual shape that satisfies the preceding rank and shape relationships and the product constraints. The two-stage ACLNN interface receives concrete tensor descriptions and does not accept `-1` or `-2` passed in as actual dimension lengths.
- For <term>Ascend 950PR&950DT products</term>, inputs and outputs in graph mode are processed as contiguous ND storage, and directly passing in tensor views with arbitrary strides or storage offsets is not supported. When calling the ACLNN interface, `x1`, `x2`, `gamma`, and the quantization parameters support non-contiguous inputs, and the interface makes them contiguous first; `y1Out`, the enabled `y2Out`, and `xOut` support non-contiguous output views, which the interface writes back. The layout conventions of the two paths apply separately.
- <term>Ascend 950PR&950DT products</term> support empty tensors with valid shapes; in this case, outputs keep the shape inference results and input elements are not read. The second quantization pass is enabled if either `scales2` or `zero_points2` exists; when only `zero_points2` exists, the scale of the second pass is treated as 1. When both are absent, `y2` in graph mode is a placeholder output with shape `[1]`, and its value is invalid; the ACLNN `y2Out` must still satisfy the shape/dtype validation of the interface, but no valid quantization result is written and it should not participate in numerical comparison.

<term>Atlas inference products</term>: The number of data elements in the dimensions of `x1` and `x2` that require normalization must not be less than 32. The number of data elements of `gamma`, `beta`, `scales1`, `scales2`, `zero_points1`, and `zero_points2` must not be less than 32.

- For <term>Ascend 950PR&950DT products</term>, the RMS reduction covers the complete tail axis corresponding to `gamma`, and the number of reduced elements is the product of these dimensions; `axis=-1` is an attribute of the quantization interface and does not narrow the RMS reduction of multi-dimensional `gamma` to the last dimension. When beta exists, its shape is consistent with gamma.
- Parameter validation failures of Kernel/GEIR are reported through the failure status and diagnostic information of graph building or tiling; 161001 and 161002 of ACLNN are return codes of that API layer and are not used as unified return codes of Kernel/GEIR.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_add_rms_norm_quant](examples/test_aclnn_add_rms_norm_quant.cpp) | Calls the AddRmsNormQuant operator through the [aclnnAddRmsNormQuant](docs/aclnnAddRmsNormQuant_en.md) interface. |
| Graph-mode invocation | -  | Calls the AddRmsNormQuant operator by building a graph through the [operator IR](op_graph/add_rms_norm_quant_proto.h).         |
