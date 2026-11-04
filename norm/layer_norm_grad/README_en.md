# LayerNormGrad

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

- Operator function: backpropagation of LayerNorm. It computes the gradients of the input tensor so that the model parameters can be updated during backpropagation.
- Formula:

  $$
  rstd = \frac{1}{\sqrt{variance + epsilon}}
  $$

  $$
  res\_for\_gamma = (input - mean) \times rstd
  $$

  $$
  dy\_g = gradOut \times weight
  $$

  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_1} gradOut \times weight
  $$

  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_1}(gradOut \times weight \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weight - (temp_1 + temp_2)) \times rstd
  $$

  $$
  gradWeightOut =  \sum_{reduce\_axis\_0}gradOut \times (input - mean) \times rstd
  $$

  $$
  gradBiasOut = \sum_{reduce\_axis\_0}gradOut
  $$

  Here, N is the dimension of the axis on which normalization is computed, that is, the size of the normalization axis dimensions.

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
      <th>Input/Output</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are supported.</li><li>Represents the gradient tensor of the backward computation, corresponding to `gradOut` in the formula.</li><li>Has the same data type as input `x`.</li><li>Has the same shape as `x`, which is [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are supported.</li><li>First input of the forward computation, corresponding to `input` in the formula.</li><li>Has the same data type as input `dy`.</li><li>Has the same shape as `dy`, which is [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are supported.</li><li>Third output of the forward computation, representing the variance of `x`, corresponding to `variance` in the formula</li>.<li>Has the same data type as input `mean` and supports only FLOAT32.</li><li>Has the same shape as `mean`, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai, consistent with the length of the axes to be normalized.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are supported.</li><li>Second output of the forward computation, representing the mean of `x`, corresponding to `mean` in the formula.</li><li>Has the same data type as input `variance` and supports only FLOAT32.</li><li>Has the same shape as `variance`, which is [A1,...,Ai,1,...,1], where there are j 1s after Ai, consistent with the length of the axes to be normalized.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td><ul><li>Empty tensors are supported.</li><li>Represents the weight tensor, corresponding to `weight` in the formula.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are supported.</li><li>Represents the output gradient of the backpropagation, corresponding to `gradInputOut` in the formula.</li><li>Has the same data type as input `x`, and its shape equals the shape of `x`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are supported.</li><li>Represents the gradient of the backpropagated weight, corresponding to `gradWeightOut` in the formula.</li><li>Has the same data type as output `pd_beta`; the data type is FLOAT32 or the same as that of input `gamma`. Its shape equals the shape of `pd_beta`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>Output</td>
      <td><ul><li>Empty tensors are supported.</li><li>Represents the gradient of the backpropagated bias, corresponding to `gradBiasOut` in the formula.</li><li>Has the same data type as output `pd_gamma`; the data type is FLOAT32 or the same as that of input `gamma`. Its shape equals the shape of `pd_gamma`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: empty tensors are not supported.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_layer_norm_grad](examples/test_geir_layer_norm_grad.cpp) | Invokes the LayerNormGrad operator by building a graph through the [Ascend IR](docs/ascendirLayerNormGrad.md). |
