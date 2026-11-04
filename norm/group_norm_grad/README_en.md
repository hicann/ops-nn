# GroupNormGrad

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

- Operator function: GroupNorm computes the group normalization result, the mean, and the reciprocal of the standard deviation of the input tensor. This operator performs the backward computation of GroupNorm. It computes the gradients of the input tensor so that model parameters can be updated during backward propagation.
- Formula:

  $$
  \hat{x} = (x - mean) \cdot rstd
  $$

  $$
  d\beta = \sum_{i=1}^n dy
  $$

  $$
  d\gamma = \sum_{i=1}^n (dy \cdot \hat{x})
  $$

  $$
  dx = rstd \cdot \gamma \begin{bmatrix}
  dy - \frac{1}{N}  (d\beta + \hat{x} \cdot d\gamma)
  \end{bmatrix}
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
      <td>Gradient tensor for backward computation, corresponding to `dy` in the formulas. The data type is the same as that of `x`. `dy` supports 2-8 dimensions (N, C, *); the computation logic focuses only on the first two dimensions (N and C), and the remaining dimensions can be merged into one dimension.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Second output of the forward computation, representing the mean of each group after `x` is grouped, corresponding to `mean` in the formulas. Must be 2D (N, num_groups).</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Third output of the forward computation, representing the reciprocal of the standard deviation of each group after `x` is grouped, corresponding to `rstd` in the formulas. The data type is the same as that of `mean`. Must be 2D (N, num_groups).</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>First input of the forward computation, corresponding to `x` in the formulas. The data type is the same as that of `dy`. Supports 2-8 dimensions (N, C, *); the computation logic focuses only on the first two dimensions (N and C), and the remaining dimensions can be merged into one dimension.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Represents the scaling coefficient of each channel, corresponding to `γ` in the formulas; the data type is the same as that of `mean`. Must be 1D. The value of `gamma` must be consistent with the C axis value of `x`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_groups</td>
      <td>Attribute</td>
      <td>Splits the C dimension of the input `dy` into group groups; group must be greater than 0.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional Attribute</td>
      <td><ul><li>Specifies the data format of the output `dx`.</li><li>Default value: NCHW.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx_is_require</td>
      <td>Optional Attribute</td>
      <td><ul><li>Whether to output `dx`.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgamma_is_require</td>
      <td>Optional Attribute</td>
      <td><ul><li>Whether to output `dgamma`.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbeta_is_require</td>
      <td>Optional Attribute</td>
      <td><ul><li>Whether to output `dbeta`.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Gradient of the computation output, used to update the gradient of the input `x`, corresponding to `dx` in the formulas. The data type is the same as that of `dy`, and the shape is the same as that of `x`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>Output</td>
      <td>Gradient of the computation output, used to update the gradient of the scaling parameter, corresponding to `dγ` in the formulas. The data type is the same as that of `mean`, and the shape is the same as that of `gamma`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>Output</td>
      <td>Gradient of the computation output, used to update the gradient of the bias parameter, corresponding to `dβ` in the formulas. The data type is the same as that of `mean`, and the shape is the same as that of `gamma`.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas training products</term>: The data types of the input and output parameters do not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_group_norm_grad](examples/test_aclnn_group_norm_grad.cpp) | Invokes the GroupNormGrad operator through the [aclnnGroupNormBackward](docs/aclnnGroupNormBackward_en.md) interface. |
| Graph-mode invocation | -  | Builds the graph and invokes the GroupNormGrad operator through the [operator IR](op_graph/group_norm_grad_proto.h).         |
