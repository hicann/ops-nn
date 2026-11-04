# LayerNorm

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function:

  Performs normalization computation with a mean of 0 and a standard deviation of 1 on the specified layer.
  - Normalization: normalizes each sample of the input tensor, so that each sample has a mean of 0 and a variance of 1.
  - Scaling and offset: after normalization, the scaled and shifted output can be further adjusted through a scaling factor and an offset, to meet the needs of different models.

- Formula:

  $$
  mean = {E}[x]
  $$

  $$
  variance = \mathrm{Var}[x]
  $$

  $$
  y = w \times (\frac{x - mean}{ \sqrt{\mathrm{Var}[x] + eps}}) + b
  $$

  In the formulas, E[x] denotes the mean of the input, and Var[x] denotes the variance of the input.

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
      <td>Denotes the input for the normalization computation, the input x in the formula. The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai denote the dimensions that do not need normalization and R1 to Rj denote the dimensions that need normalization.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the weight for the normalization computation, the input w in the formula. gamma is a required input and cannot be empty. Its data type is the same as input x or FLOAT, and is the same as the data type of beta. The shape is [R1,...,Rj], [Rn,...,Rj], or [Am,...,Ai,R1,...,Rj] (1 < n ≤ j, 1 ≤ m ≤ i), the same as the consecutive-dimension shape of input x starting from begin_params_axis.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Denotes the offset for the normalization computation, the input b in the formula. beta is a required input and cannot be empty. Its data type is the same as input x or FLOAT, and is the same as the data type of gamma. Its shape matches the shape of gamma.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>begin_norm_axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the dimension on which the norm computation is performed. This dimension must be within the shape dimension range of input x.</li><li>Default value: 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>begin_params_axis</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the index of the start dimension of input gamma or beta. This dimension must be within the shape dimension range of input x.</li><li>Default value: 0.</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>The value added to the denominator to ensure numerical stability.</li><li>Default value: 1e-12.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the result of the normalization computation, the y in the formula. Its shape must equal the shape of x, that is [A1,...,Ai,R1,...,Rj]. Its data type matches the data type of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Denotes the mean after normalization, the mean in the formula. Its shape matches the shape of variance, that is [A1,...,Ai,1,...,1], where there are j ones after Ai, consistent with the length of the axes that need normalization. Its data type matches the data types of gamma and beta.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Output</td>
      <td>Denotes the variance after normalization, the variance in the formula. Its shape matches the shape of mean, that is [A1,...,Ai,1,...,1], where there are j ones after Ai, consistent with the length of the axes that need normalization. Its data type matches the data types of gamma and beta.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_layer_norm](./examples/test_geir_layer_norm.cpp) | Calls the LayerNorm operator by building the graph through [Ascend IR](./docs/ascendirLayerNorm.md). |
