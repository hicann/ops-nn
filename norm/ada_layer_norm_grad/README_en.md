# AdaLayerNormGrad

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

- Operator function: the backward propagation of [AdaLayerNormV2](../ada_layer_norm_v2/README.md). It computes the gradients of the input tensors so that model parameters can be updated during backward propagation.
- Computation formula:

  Forward:

  $$
  out = LayerNorm(x)\times(1+scale)+shift
  $$

  LayerNorm computation formula:

  $$
  LayerNorm(x) = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * weightOptional + biasOptional
  $$

  where E(x) denotes the mean of the input, and Var(x) denotes the variance of the input.

  Backward:

  $$
  z = (x - mean) \times rstd
  $$

  $$
  dy\_g = dy \times gamma \times (1 + scale)
  $$

  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_2} (dy \times gamma \times (1 + scale))
  $$

  $$
  temp_2 = 1/N \times (x - mean) \times rstd \times \sum_{reduce\_axis\_2}(dy \times gamma \times (1 + scale) \times (x - mean) \times rstd)
  $$

  $$
  pd\_x = (dy \times gamma \times (1 + scale) - (temp_1 + temp_2)) \times rstd
  $$

  $$
  pd\_scale =  \sum_{reduce\_axis\_1}(dy \times ((x - mean) \times rstd \times gamma + beta))
  $$

  $$
  pd\_shift =  \sum_{reduce\_axis\_1}dy
  $$

  $$
  pd\_gamma =  \sum_{reduce\_axis\_0,1}dy \times (1 + scale) \times (x - mean) \times rstd
  $$

  $$
  pd\_beta = \sum_{reduce\_axis\_0,1}dy \times (1 + scale)
  $$

  where N is the dimensions of the axis on which the normalization computation is performed, that is, the size of the normalization axis dimensions; 'reduce_axis_0,1' indicates that reduction and summation are performed on dimension 0 and dimension 1.

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
      <td>Gradient tensor of the backward computation, corresponding to `dy` in the computation formula. It has the same data type as input `x`. Its shape is equal to the shape of `x`, which is [B..., S, H], where B supports 0 to 6 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>First input of the forward computation, corresponding to `x` in the computation formula. It has the same data type as input `dy`. Its shape is equal to the shape of `dy`, which is [B..., S, H], where B supports 0 to 6 dimensions.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Denotes the reciprocal of the standard deviation of `x`, corresponding to `rstd` in the computation formula. The data type is fixed to FLOAT32. Its shape is equal to the shape of `mean`, which is [B..., S, 1], where the last dimension is fixed to 1 and the other dimensions are consistent with `x`.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Denotes the mean of `x`, corresponding to `mean` in the computation formula. The data type is fixed to FLOAT32. Its shape is equal to the shape of `rstd`, which is [B..., S, 1].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Denotes the adaptive scaling tensor, corresponding to `scale` in the formula. Its data type is the same as the data type of `dy`. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `dy`, and H is consistent with the H dimension in `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the weight tensor, corresponding to `gamma` in the formula. The data types FLOAT32, FLOAT16, and BFLOAT16 are supported; when the data type is not FLOAT32, the parameter data type is the same as `dy`. The shape is [H], where H is consistent with the H dimension in `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Denotes the bias tensor, corresponding to `beta` in the formula. Its data type is the same as the data type of `gamma`. The shape is [H], where H is consistent with the H dimension in `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>Output</td>
      <td>Denotes the output gradient of the backward propagation, corresponding to `pd_x` in the computation formula. It has the same data type as input `x`, and its shape is equal to the shape of `x`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_scale</td>
      <td>Output</td>
      <td>Denotes the gradient of the adaptive scaling coefficient in the backward propagation, corresponding to `pd_scale` in the computation formula. It has the same data type as input `scale`, and its shape is equal to the shape of `scale`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_shift</td>
      <td>Output</td>
      <td>Denotes the gradient of the adaptive offset coefficient in the backward propagation, corresponding to `pd_shift` in the computation formula. It has the same data type as input `scale`. The shape is [B..., H] or [B..., 1, H], where B supports 0 to 6 dimensions, the number and sizes of the dimensions are consistent with B in `dy`, and H is consistent with the H dimension in `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>Output</td>
      <td>Denotes the gradient of the weight in the backward propagation, corresponding to `pd_gamma` in the computation formula. It has the same data type as input `gamma`, and its shape is equal to the shape of `gamma`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>Output</td>
      <td>Denotes the gradient of the bias in the backward propagation, corresponding to `pd_beta` in the computation formula. It has the same data type as input `gamma`. Its shape is equal to the shape of `pd_gamma`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The number of rows of inputs `dy` and `x` (the product of batch and seq) and the number of columns (the H dimension) must both be greater than or equal to 1.
- The number of columns H is limited by the UB capacity. Extremely large columns (such as 8K or above) are supported by the cutD blocking path, but the width of a single block is limited by the available UB size.
- The maximum dimensions supported by the aclnn interface are 8.
- Mixed precision is supported: `dy`/`x`/`scale`/`shift` support FLOAT32, FLOAT16, and BFLOAT16; `rstd`/`mean` are fixed to FLOAT32; `gamma`/`beta` support FLOAT32, FLOAT16, and BFLOAT16 (when the type is not FLOAT32, it is the same as the type of `dy`).
- Empty tensors are not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_ada_layer_norm_grad](examples/test_aclnn_ada_layer_norm_grad.cpp) | Calls the AdaLayerNormGrad operator through the [aclnnAdaLayerNormBackward](docs/aclnnAdaLayerNormBackward_en.md) interface. |
| Graph mode | -  | Calls the AdaLayerNormGrad operator by constructing a graph through the [operator IR](op_graph/ada_layer_norm_grad_proto.h).         |
