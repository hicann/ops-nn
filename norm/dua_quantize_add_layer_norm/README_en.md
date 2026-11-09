# DuaQuantizeAddLayerNorm

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: DuaQuantizeAddLayerNorm performs a complex computation that combines the functions of quantization, addition, and layer normalization (Layer Normalization).

- Formula:

  $$
  x = x1 + x2 + bias
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+epsilon}}
  $$

  $$
  norm\_out = (x-\bar{x}) * rstd * gamma + beta
  $$

  - When axis=-65535:

    $$
    out1 = round((norm\_out * scales1)+zero\_points1)
    $$

    $$
    out2 = round((norm\_out * scales2)+zero\_points2)
    $$

  - When axis≠-65535:

    $$
    out1 = round((norm\_out/scales1)+zero\_points1)
    $$

    $$
    out2 = round((norm\_out/scales2)+zero\_points2)
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
      <td>x1</td>
      <td>Input</td>
      <td>Input x1 of the forward fusion operator, corresponding to `x1` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input x2 of the forward fusion operator, corresponding to `x2` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight used for the normalization computation, corresponding to `gamma` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Offset used in normalization, corresponding to `beta` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input of the addition computation in DuaQuantizeAddLayerNorm, corresponding to `bias` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>Scale factor used to quantize the first input tensor, corresponding to `scales1` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Input</td>
      <td>Scale factor used to quantize the second input tensor, corresponding to `scales2` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>Offset used to quantize the first input tensor, corresponding to `zero_points1` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>Offset used to quantize the second input tensor, corresponding to `zero_points2` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>Attribute</td>
      <td><ul><li>Specifies the data type of the quantized output tensor y.</li><li>There is no default value.</li><li>Currently, only INT8 is delivered (y1 and y2 are fixed to INT8).</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether the static quantization processes scale by multiplication or division. When axis is -65535, multiplication is used; for other values of axis, division is used.</li><li>The default value is -1.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to `epsilon` in the formula.</li><li>The default value is 1e-5.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether to enable the output of x (reserved switch).</li><li>The default value is false.</li><li>In the current implementation, x is a required output and is always written; this attribute does not affect the output of x.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>First quantized output tensor, corresponding to `out1` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Second quantized output tensor, corresponding to `out2` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Data types: the dtypes of the 9 inputs must be consistent, taking the values FLOAT16, BFLOAT16, and FLOAT32.
- Dimensions: the rank of x1 (and x2) ranges from [1, 8]; a scalar with rank 0 is processed as shape [1].
- Normalization dimension: the last dimension of x1 (W = x1.shape[-1]) must be greater than 0.
- Input shape: x2.shape must equal x1.shape.
- Parameter shapes: gamma, beta, bias, scales1, scales2, zero_points1, and zero_points2 must be 1-dimensional, and their lengths must equal W.
- Outputs: the shapes of y1, y2, and x must equal x1.shape; the dtypes of y1 and y2 are fixed to INT8, and the dtype of x is consistent with x1.
- Quantization parameters: when axis is not -65535 (division mode), scales1 and scales2 must not be 0; the implementation does not guard against division by zero, and results propagate according to Inf/NaN semantics.
- Value range: no value range check is performed on epsilon; when the square term of Var(x)+epsilon exceeds the representable range of FLOAT32, the result propagates according to IEEE semantics (for example, when the variance square overflows to Inf, rstd is 0 and norm_out degrades to beta).
- Platform differences: for Atlas A2 (ascend910b) and Atlas A3 (ascend910_93), only BFLOAT16 binaries are currently delivered; FLOAT16 and FLOAT32 are delivered only on the Ascend 950 series.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>, all input parameters and output parameter `x` only support the BFLOAT16 data type.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_dua_quantize_add_layer_norm](examples/test_geir_dua_quantize_add_layer_norm.cpp)  | Invokes the DuaQuantizeAddLayerNorm operator through graph construction with the [operator IR](op_graph/dua_quantize_add_layer_norm_proto.h).         |
