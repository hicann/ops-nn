# LayerNormV4

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×   |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function:

  Performs normalization computation with a mean of 0 and a standard deviation of 1 on the specified layer.
  - Normalization: normalizes each sample of the input tensor so that each sample has a mean of 0 and a variance of 1.
  - Scaling and offset: after normalization, you can further adjust the normalized output with a scaling factor and an offset to meet the requirements of different models.

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

  Here E[x] denotes the mean of the input, and Var[x] denotes the variance of the input.

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
      <td>Denotes the input for the normalization computation, corresponding to `x` in the formula. The shape is [A1,...,Ai,R1,...,Rj], where A1 to Ai denote the dimensions that do not need norm, and R1 to Rj denote the dimensions that need norm.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>normalized_shape</td>
      <td>Input</td>
      <td>Denotes the dimensions that need norm computation. The value is [R1,...,Rj], with a length less than or equal to the shape length of the input x; an empty value is not supported.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Optional input</td>
      <td>Denotes the weight for the normalization computation, the input `w` in the formula. When gamma is not empty, its data type is the same as the input x or is FLOAT, and when beta exists, gamma and beta have the same data type. The shape equals normalized_shape, that is, [R1,...,Rj]. When gamma is empty, the interface internally constructs a tensor with the shape [R1,...,Rj] and all values of 1; when beta exists, gamma and beta have the same data type, and when beta does not exist, gamma has the same data type as the input x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Optional input</td>
      <td>Denotes the offset for the normalization computation, the input `b` in the formula. When beta is not empty, its data type is the same as the input x or is FLOAT, and when gamma exists, beta and gamma have the same data type. The shape equals normalized_shape, that is, [R1,...,Rj]. When beta is empty, the interface internally constructs a tensor with the shape [R1,...,Rj] and all values of 0; when gamma exists, beta and gamma have the same data type, and when gamma does not exist, beta has the same data type as the input x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>The value added to the denominator to ensure numerical stability, corresponding to eps in the formula.</li><li>The default value is 0.00001f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Denotes the result of the normalization computation, the output `y` in the formula. The shape must equal the shape of x, that is, [A1,...,Ai,R1,...,Rj]. The data type is the same as the data type of x.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Output</td>
      <td>Denotes the mean after normalization, corresponding to `mean` in the formula. It has the same shape as rstd, and the shape is [A1,...,Ai,1,...,1], with j ones after Ai, consistent with the length of the axes that need norm. The data type is FLOAT32.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Output</td>
      <td>Denotes the reciprocal of the standard deviation after normalization, corresponding to `rstd` in the formula. It has the same shape as mean, and the shape is [A1,...,Ai,1,...,1], with j ones after Ai, consistent with the length of the axes that need norm. The data type is FLOAT32.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term>: x and y support only FLOAT16; gamma/beta support FLOAT and FLOAT16.

  - <term>Kirin X90 processor series products</term> and <term>Kirin 9030 processor series products</term>:
    - The data types of all inputs and outputs do not support BFLOAT16.
    - The data type of normalized_shape does not support INT64.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_fast_layer_norm](examples/test_aclnn_fast_layer_norm.cpp) | Calls the LayerNormV4 operator through the [aclnnFastLayerNorm](docs/aclnnFastLayerNorm_en.md) interface. |
| aclnn API  | [test_aclnn_layer_norm_v4](examples/test_aclnn_layer_norm_v4.cpp) | Calls the LayerNormV4 operator through the [aclnnLayerNorm](docs/aclnnLayerNorm&aclnnLayerNormWithImplMode_en.md) interface. |
| aclnn API  | [test_aclnn_layer_norm_with_impl_mode](examples/test_aclnn_layer_norm_with_impl_mode.cpp) | Calls the LayerNormV4 operator through the [aclnnLayerNormWithImplMode](docs/aclnnLayerNorm&aclnnLayerNormWithImplMode_en.md) interface. |
| GE graph mode | [test_geir_layer_norm_v4](examples/test_geir_layer_norm_v4.cpp) | Calls the LayerNormV4 operator by building a graph through [Ascend IR](docs/ascendirLayerNormV4.md). |
