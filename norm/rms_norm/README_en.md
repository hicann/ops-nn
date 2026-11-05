# RmsNorm

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
|  <term>KirinX90 processor series products</term>       |     √    |
|  <term>Kirin9030 processor series products</term> | √ |

## Function

- Operator function: the RmsNorm operator is a normalization operation commonly used by large models. Compared with the LayerNorm operator, it removes the mean subtraction part.
- Formula:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
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
      <td>Input for the normalization computation. It is `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling factor (weight) for the normalization computation, that is, `g` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability, corresponding to eps in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Final output after normalization, that is, `RmsNorm(x)` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Reciprocal of the standard deviation after normalization, that is, the reciprocal of `Rms(x)` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>, <term>KirinX90 processor series products</term>, and <term>Kirin9030 processor series products</term>: the data types of all input parameters and output parameters do not support BFLOAT16.

## Constraints

<term>Atlas inference products</term>: the data volume of the normalization dimensions [R1,...,Rj] (R1×...×Rj × the byte count of a single element) must be greater than or equal to 32 Bytes.

<term>Ascend 950PR&950DT products</term>: when the number of elements of the normalization suffix is 0 and the number of elements of the non-normalized dimensions is greater than 0, `x`, `gamma`, and `y` are empty tensors, and `rstd` is non-empty with all elements being NaN; when the number of elements of the non-normalized dimensions is 0, both `y` and `rstd` are empty tensors. The shape of `gamma` must still be consistent with the normalization suffix of `x`.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_rms_norm](examples/test_aclnn_rms_norm.cpp) | Calls the RmsNorm operator through the [aclnnRmsNorm](docs/aclnnRmsNorm_en.md) interface. |
| Graph mode | [test_geir_rms_norm](examples/test_geir_rms_norm.cpp) | Calls the RmsNorm operator by constructing a graph through [Ascend IR](docs/ascendir_RmsNorm.md). |
