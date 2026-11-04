# RmsNormGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Backward computation of [RmsNorm](../rms_norm/README.md). It computes the gradient of RmsNorm, that is, computes the gradient of the input tensor during backpropagation.
- Operator formula:

  - Forward formula:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  - Backward derivation:

  $$
  dx_i= (dy_i * g_i - \frac{x_i}{\operatorname{Rms}(\mathbf{x})} * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * \frac{x_i}{\operatorname{Rms}(\mathbf{x})})
  $$

  $$
  dg_i = \frac{x_i}{\operatorname{Rms}(\mathbf{x})} dy_i
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
      <td>Gradient passed back, corresponding to `dy` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input of the forward operator, representing the data to be normalized, corresponding to `x` in the formula. The shape is the same as the shape of the input parameter `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Intermediate computation result of the forward operator, corresponding to the reciprocal of `Rms(x)` in the formula. The shape must satisfy rstd_shape = x_shape\[0:n\], where n < x_shape.dims() and n is consistent with gamma.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling tensor in the normalization process, corresponding to `g` in the formula. The shape must satisfy gamma_shape = x_shape\[n:\], where n < x_shape.dims().</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Gradient of the input `x`, corresponding to `dx` in the formula. The shape must be the same as the input `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>Output</td>
      <td>Gradient of `gamma`, corresponding to `dg` in the formula. The shape must be the same as the input `gamma`.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: The data types of the input parameters `dy`, `x`, and `gamma` and the output `dx` do not support BFLOAT16.

## Constraints

<term>Atlas inference products</term>: The data volume of the normalization dimensions [R1,...,Rj] (R1×...×Rj× the number of bytes of a single element) must be greater than or equal to 32 Bytes.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_rms_norm_grad](examples/test_aclnn_rms_norm_grad.cpp) | Invoke the RmsNormGrad operator through the [aclnnRmsNormGrad](docs/aclnnRmsNormGrad_en.md) API. |
| Graph mode | [test_geir_rms_norm_grad](examples/test_geir_rms_norm_grad.cpp) | Invoke the RmsNormGrad operator by building a graph through the [Ascend IR](docs/ascendirRmsNormGrad.md). |
