# InplaceAddRmsNorm

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |
| <term>Kirin X90 processor series products</term> | √ |
| <term>Kirin 9030 processor series products</term> | √ |

## Function

- Description: The RmsNorm operator is a normalization operation commonly used in large models. Compared with the LayerNorm operator, it removes the mean subtraction part. The AddRmsNorm operator fuses the Add operator before RmsNorm to reduce data move-in and move-out operations. InplaceAddRmsNorm is an operation that combines in-place addition and RMS normalization.
- Formula:

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  \operatorname{RmsNorm}(x_i)= g_i * (x_i * \operatorname{rstd}(\mathbf{x})), \quad \text { where } \operatorname{rstd}(\mathbf{x})=\frac{1}{\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}}
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
      <td>The first input for the Add computation, corresponding to `x1` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The second input for the Add computation, corresponding to `x2` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the scaling factor (weight) of RmsNorm, corresponding to `g` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>A value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to `eps` in the formula.</li><li>Default value: 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Output</td>
      <td>Denotes the final output, an aclTensor on the Device side, corresponding to `RmsNorm(x)` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Denotes the reciprocal of the standard deviation after normalization, corresponding to `rstd` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Output</td>
      <td>Denotes the result of the Add computation, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>:
    - The data types of all input parameters and the output parameters `x1` and `x2` do not support BFLOAT16.
    - In the usage scenario on the current product, the output parameter `rstd` is an invalid parameter, and the output value does not take effect.

  - <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>: the data types of all inputs and outputs do not support BFLOAT16.

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_inplace_add_rms_norm](examples/test_aclnn_inplace_add_rms_norm.cpp) | Invokes the InplaceAddRmsNorm operator through the [aclnnInplaceAddRmsNorm](docs/aclnnInplaceAddRmsNorm_en.md) interface. |
| Graph mode | -  | Invokes the InplaceAddRmsNorm operator by building a graph through the [operator IR](op_graph/inplace_add_rms_norm_proto.h).         |
