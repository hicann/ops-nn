# GemmaRmsNorm

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: The GemmaRmsNorm operator is a normalization operation commonly used in large models. Compared with the RmsNorm operator, it adds 1 to gamma during computation.
- Formula:

  $$
  \operatorname{GemmaRmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} (1 + g_i), \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
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
      <td><ul><li>Data input to be normalized, corresponding to `x` in the formula.</li><li>The shape supports 1D to 8D.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td><ul><li>Data scaling factor, corresponding to `g` in the formula.</li><li>The data type is the same as the data type of input `x`.</li><li>The shape supports 1D to 8D and satisfies gamma_shape = x_shape[n:], n < x_shape.dims().</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to `eps` in the formula.</li><li>Default value: 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td><ul><li>Normalized output data, corresponding to `GemmaRmsNorm(x)` in the formula.</li><li>The data type and shape are the same as those of input `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td><ul><li>Reciprocal of the standard deviation of x, corresponding to the reciprocal of `Rms(x)` in the formula.</li><li>The shape supports 1D to 8D and is consistent with the leading dimensions of the shape of input parameter `x`. The leading dimensions refer to the dimensions of `x` minus the dimensions of `gamma`, indicating the dimensions that do not need normalization.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: BFLOAT16 is not supported for the data types of x, gamma, and y.

## Constraints

None

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_gemma_rms_norm](examples/test_aclnn_gemma_rms_norm.cpp) | Invoke the GemmaRmsNorm operator through the [aclnnGemmaRmsNorm](docs/aclnnGemmaRmsNorm_en.md) API. |
| Graph mode | -  | Invoke the GemmaRmsNorm operator through graph construction using the [operator IR](op_graph/gemma_rms_norm_proto.h).         |
