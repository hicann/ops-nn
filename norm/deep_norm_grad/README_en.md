# DeepNormGrad

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

- Description: Backward propagation of [DeepNorm](../deep_norm/README.md). It completes the gradient computation of tensor x, tensor gx, and tensor gamma, and the summation computation of tensor dy.

- Formula:

  $$
  dgx_i = tmpone_i * rstd + dvar * tmptwo_i + dmean
  $$

  $$
  dx_i = alpha * {dgx}_i
  $$

  $$
  dbeta = \sum_{i=1}^{N} dy_i
  $$

  $$
  dgamma =  \sum_{i=1}^{N} dy_i * rstd * {tmptwo}_i
  $$

  where:

  $$
  oneDiv=-1/SizeOf(gamma)
  $$

  $$
  tmpone_i = dy_i * gamma
  $$

  $$
  tmptwo_i = alpha * x_i + {gx}_i - mean
  $$

  $$
  dvar = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * {tmptwo}_i * {rstd}^3
  $$

  $$
  dmean = (oneDiv) * \sum_{i=1}^{N} {tmpone}_i * rstd
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
      <td>The main grad input, corresponding to `dy` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>The input x of the forward fusion operator, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>Input</td>
      <td>The input gx of the forward fusion operator, corresponding to `gx` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>The scaling parameter of forward propagation, corresponding to `gamma` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Denotes the mean of the sum of the forward inputs x and gx, corresponding to `mean` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Denotes the rstd of the sum of the forward inputs x and gx, corresponding to `rstd` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Optional attribute</td>
      <td><ul><li>The meaning is the same as the forward input alpha of deepnorm. It is the multiplier weight parameter of the x dimension input of deepnorm, that is, the input `alpha` in the formula.</li><li>Default value: 0.3f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>The gradient of the computed output, used to update the gradient of the input data x, corresponding to `dx` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgx</td>
      <td>Output</td>
      <td>The gradient of the computed output, used to update the gradient of the input data gx, corresponding to `dgx` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>Output</td>
      <td>The gradient of the computed output, used to update the gradient of the bias parameter, corresponding to `dbeta` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>Output</td>
      <td>The gradient of the computed output, used to update the gradient of the scaling parameter. It corresponds to `dgamma` in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: the data types of the input parameters x, gx, and gamma and the output parameters dx and dgx do not support BFLOAT16.

## Constraints

- The shapes of `dy`, `x`, `gx`, `dx`, and `dgx` must be exactly the same, and all dimensions of `dy` must be greater than 0.
- The ranks of `mean` and `rstd` must be consistent with that of `dy`, the leading dimensions must be aligned with `dy`, the trailing dimensions corresponding to normalization must all be `1`, and the shapes of `mean` and `rstd` must be consistent.
- The rank of `gamma` must be less than the rank of `dy`, and each of its dimensions must match the trailing dimensions of `dy` one by one; the shapes of `dbeta` and `dgamma` must also be consistent with that of `gamma`.
- The dtypes of `dy/x/gx/gamma/dx/dgx` must be consistent, and only `FLOAT32`, `FLOAT16`, and `BFLOAT16` are supported; `mean/rstd/dbeta/dgamma` support only `FLOAT32`.
- The tiling side performs overflow checks on the shape product and byte span, and directly rejects the input when the constraints are not met.
- When `numCols < 500 && numRows >= 1024`, the operator uses the small-D tiling branch and requires additional workspace; other scenarios use the normal branch.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_deep_norm_grad](examples/test_aclnn_deep_norm_grad.cpp) | Invokes the DeepNormGrad operator through the [aclnnDeepNormGrad](docs/aclnnDeepNormGrad_en.md) interface. |
| Graph mode | -  | Invokes the DeepNormGrad operator by building a graph through the [operator IR](op_graph/deep_norm_grad_proto.h).         |
