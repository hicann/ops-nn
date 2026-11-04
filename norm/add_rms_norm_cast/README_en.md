# AddRmsNormCast

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
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: the RmsNorm operator is a common normalization operation for large models. The AddRmsNormCast operator fuses the Cast operator after AddRmsNorm, reducing data copy-in and copy-out operations.
- Formula:

  $$
  x_i=x1_{i}+x2_{i}
  $$

  $$
  y_2=\operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

  $$
  y_1=float(y_2)
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
      <td>The raw data input to be normalized, the input `x1` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The raw data input to be normalized, the input `x2` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>The data scaling factor, the input `g` in the formula. The shape must be consistent with the last several dimensions of `x1`, which are the dimensions of x1 that need norm.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>The value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to eps in the formula.</li><li>The default value is 1e-6.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>The output data after type conversion following normalization, the output `y1` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>The output data after normalization, the output `y2` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>The standard deviation of x, the output `Rms(x)` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>The sum of the data for normalization, the output `x` in the formula.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- For <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>, the data types of x1, x2, gamma, y2, and x do not support BFLOAT16.

## Constraints

The output does not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_add_rms_norm_cast](examples/test_aclnn_add_rms_norm_cast.cpp) | Invokes the AddRmsNormCast operator through the [aclnnAddRmsNormCast](docs/aclnnAddRmsNormCast_en.md) API. |
| Graph mode | -  | Invokes the AddRmsNormCast operator through graph construction using the [operator IR](op_graph/add_rms_norm_cast_proto.h).         |
