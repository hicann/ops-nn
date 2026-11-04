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

- Operator function: Performs deep normalization on the elements of input tensor x. By computing the mean and standard deviation, it normalizes each element into an output tensor with zero mean and unit variance. Specifically, the input tensor `x` is first multiplied by the weight `alpha` and then added to `gx` to obtain the intermediate tensor `x'`. Then the mean `mean` and the reciprocal of the standard deviation `rstd` of `x'` are computed along the normalization dimensions to complete the normalization, and the output tensor `y` is obtained through `gamma` scaling and `beta` bias. The operator outputs `mean`, `rstd`, and `y` at the same time.
- Formula:

  $$
  \begin{aligned}
  x_i^{\prime} &= \mathrm{alpha} \cdot x_i + \mathrm{gx}_i \\
  \mathrm{mean} &= \frac{1}{n}\sum_{i=1}^{n}x_i^{\prime} \\
  \mathrm{rstd} &= \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i^{\prime} - \mathrm{mean})^2 + \mathrm{epsilon}}} \\
  y_i &= (x_i^{\prime} - \mathrm{mean}) \cdot \mathrm{rstd} \cdot \mathrm{gamma}_i + \mathrm{beta}_i
  \end{aligned}
  $$

  where the normalization dimensions are the trailing dimensions of `x` corresponding to the shape of `gamma`, and `n` is the product of the sizes of these dimensions. The preceding formulas apply to a group of data with a fixed leading dimension index, and `i` is the element index of the group after the normalization dimensions are flattened. `gamma` and `beta` are shared across different groups. `x'` is the intermediate tensor in the computation process, `mean` and `rstd` are the statistics of each group of data, `y` is the final output, and `epsilon` is the numerical stability term added to the variance.

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
      <td>Empty tensors are not supported. The input data that is added element-wise to the weighted `x`, corresponding to `gx` in the formula. The data type is the same as that of input `x`, and the shape is the same as that of input `x`.</td>
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
      <td>The numerical stability term added to the variance of `x'` when computing `rstd`, used to prevent division-by-zero errors, corresponding to `epsilon` in the formula. The default value is 1e-06f.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The mean computed for the intermediate tensor `x'` along the normalization dimensions, corresponding to the output `mean` in the formula, used to compute `rstd` and `y`. The shape satisfies the broadcast relationship with input `x` (the leading dimensions (the dimensions that do not require normalization) are the same as those of input `x`, and the remaining dimensions are 1).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The reciprocal obtained by computing the variance of the intermediate tensor `x'` along the normalization dimensions, adding `epsilon`, and taking the square root, corresponding to the output `rstd` in the formula, used to compute `y`. The shape satisfies the broadcast relationship with input `x` (the leading dimensions (the dimensions that do not require normalization) are the same as those of input `x`, and the remaining dimensions are 1).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Empty tensors are not supported. The result of the intermediate tensor `x'` after normalization, `gamma` scaling, and `beta` bias, corresponding to the output `y` in the formula. The data type is the same as that of input `x`, and the shape is the same as that of input `x`.</td>
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
