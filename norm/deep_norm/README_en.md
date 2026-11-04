# DeepNorm

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Performs deep normalization on the elements of input tensor x. By computing the mean and standard deviation, it normalizes each element into an output tensor with zero mean and unit variance.
- Formula:

  $$
  DeepNorm(x_i^{\prime}) = (x_i^{\prime} - \bar{x^{\prime}}) * rstd * gamma + beta,
  $$

  $$
  \text { where } rstd = \frac{1}{\sqrt{\frac{1}{n} \sum_{i=1}^n (x^{\prime}_i - \bar{x^{\prime}})^2 + eps}} , \quad x^{\prime}_i = alpha * x_i + gx_i
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
      <td>Empty tensors are not supported. Input data, usually the output of an intermediate layer of a neural network, input `x` in the formula. The shape supports 2-8 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>Input</td>
      <td>Empty tensors are not supported. The gradient of the input data, used for backpropagation, input `gx` in the formula. The data type is the same as that of input `x`, and the shape is the same as that of input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Bias parameter, used to adjust the normalized output, input `beta` in the formula. The data type is the same as that of input `x`, the number of dimensions is the same as that of input `gamma`, and the shape is consistent with the trailing dimensions of input `x` (the dimensions that require normalization).</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Empty tensors are not supported. Scale parameter, used to adjust the normalized output, input `gamma` in the formula. The data type is the same as that of input `x`, the number of dimensions ranges over [1, 7] and must be less than the number of dimensions of input `x`, and the shape is consistent with the trailing dimensions of input `x` (the dimensions that require normalization).</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Optional attribute</td>
      <td>Weight parameter, used to adjust the weight of input x, input `alpha` in the formula. The default value is 0.3f.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td>Value added to the denominator to ensure numerical stability and to prevent division-by-zero errors, corresponding to eps in the formula. The default value is 1e-06f.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The computed output mean, used for the normalization operation, output `mean` in the formula. The shape satisfies the broadcast relationship with input `x` (the leading dimensions (the dimensions that do not require normalization) are the same as those of input `x`, and the remaining dimensions are 1).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The computed reciprocal of the standard deviation, used for the normalization operation, output `rstd` in the formula. The shape satisfies the broadcast relationship with input `x` (the leading dimensions (the dimensions that do not require normalization) are the same as those of input `x`, and the remaining dimensions are 1).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The normalized output data, output `y` in the formula. The data type is the same as that of input `x`, and the shape is the same as that of input `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: The data types of all input parameters and the output parameter y do not support BFLOAT16.

  - <term>Kirin X90 processor series products</term>, <term>Kirin 9030 processor series products</term>: The data types of all input parameters and the output parameter y do not support BFLOAT16.

## Constraints

- The number of dimensions of `x` supports 2-8, and the number of dimensions of `gamma` supports 1-7 and must be less than the number of dimensions of `x`.
- The dtypes of `x`, `gx`, `beta`, `gamma`, and `y` must be the same and support only `FLOAT32`, `FLOAT16`, and `BFLOAT16`; `mean` and `rstd` are fixed to `FLOAT32`.
- The shapes of `gx` and `y` must be consistent with `x`; the shapes of `beta` and `gamma` must be consistent with the trailing dimensions of `x`.
- The shapes of `mean` and `rstd` must be consistent with the leading dimensions of `x`, the trailing dimensions corresponding to normalization must all be `1`, and the shapes of `mean` and `rstd` must be consistent.
- Each dimension of `x` must be greater than 0. The tiling side performs range checks on `numRow/numColAlign` and the product of the shape, and inputs that exceed `uint32` are rejected.
- arch35 tiling switches between the full-load and partial-load paths based on the reduce dimension length and the UB space; the partial-load scenario requires additional workspace.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_deep_norm](examples/test_aclnn_deep_norm.cpp) | Calls the DeepNorm operator through the [aclnnDeepNorm](docs/aclnnDeepNorm_en.md) interface. |
| Graph-mode invocation | -  | Calls the DeepNorm operator by building a graph through the [operator IR](op_graph/deep_norm_proto.h).         |
