# LayerNormV3

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×   |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  Performs normalization computation with a mean of 0 and a standard deviation of 1 on the specified layer.
  - Normalization: normalizes each sample of the input tensor so that each sample has a mean of 0 and a variance of 1.
  - Scaling and offset: after normalization, the normalized output can be further adjusted using the scaling factor and the offset to meet different model requirements.

- Formula:

  $$
  mean = {E}[x]
  $$

  $$
  rstd = \frac{1}{ \sqrt{\mathrm{Var}[x] + eps}}
  $$

  $$
  y = w*((x - mean) * rstd) + b
  $$

  Here, E[x] is the mean of the input, and Var[x] is the variance of the input.

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
      <td>Represents the input for the normalization computation, the input x in the formula. The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai are the dimensions that do not need norm, and R1 to Rj are the dimensions that need norm.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Represents the weight for the normalization computation, the input w in the formula. gamma is a required input and does not support being empty. The data type is consistent with the input x or is FLOAT, and is the same as the data type of beta. The shape is [R1,...,Rj], [Rn,...,Rj], or [Am,...,Ai,R1,...,Rj] (1 < n ≤ j, 1 ≤ m ≤ i), the same as the shape of the consecutive dimensions of the input x starting from begin_params_axis.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Represents the offset for the normalization computation, the input b in the formula. beta is a required input and does not support being empty. The data type is consistent with the input x or is FLOAT, and is the same as the data type of gamma. The shape is the same as the shape of gamma.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin_norm_axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the dimension for the norm computation. This dimension must be within the shape dimension range of the input x.</li><li>The default value is 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>begin_params_axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Represents the index value of the start dimension of the input gamma or beta. This dimension must be within the shape dimension range of the input x.</li><li>The default value is 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>The value added to the denominator to ensure numerical stability.</li><li>The default value is 0.00001f.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Represents the result of the normalization computation, the y in the formula. The shape must be equal to the shape of x, that is, [A1,...,Ai,R1,...,Rj]. The data type is consistent with the data type of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Represents the mean after normalization, the mean in the formula. The shape is the same as that of rstd, that is, [A1,...,Ai,1,...,1], with j ones after Ai, consistent with the length of the axes that need norm. The data type is consistent with the data types of gamma and beta.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation after normalization, the rstd in the formula. The shape is the same as that of mean, that is, [A1,...,Ai,1,...,1], with j ones after Ai, consistent with the length of the axes that need norm. The data type is consistent with the data types of gamma and beta.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_layer_norm_v3](./examples/test_geir_layer_norm_v3.cpp)   | Invokes the LayerNormV3 operator through graph construction using [Ascend IR](./docs/ascendirLayerNormV3.md). |
