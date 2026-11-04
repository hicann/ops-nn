# LayerNormGradV3

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

- Operator function: the backward propagation of [LayerNormV4](../layer_norm_v4/README.md). It computes the gradients of the input tensors so that model parameters can be updated during backward propagation.
- Computation formula:

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

  where N is the dimensions of the axis on which the normalization computation is performed, that is, the size of the normalization axis dimensions.

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
      <td>Gradient tensor of the backward computation, corresponding to `gradOut` in the computation formula. It has the same data type as input x. Its shape is equal to the shape of x, which is [A1,...,Ai,R1,...,Rj]. Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>First input of the forward computation, corresponding to `input` in the computation formula. It has the same data type as input `dy`. Its shape is equal to the shape of `dy`, which is [A1,...,Ai,R1,...,Rj]. Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Third output of the forward computation, denoting the reciprocal of the standard deviation of `x`, corresponding to `rstd` in the computation formula. It has the same data type as input `mean`, and only FLOAT32 is supported. Its shape is equal to the shape of `mean`, which is [A1,...,Ai,1,...,1], where there are j ones after Ai, consistent with the length of the axes to be normalized. Empty tensors are supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Second output of the forward computation, denoting the mean of `x`, corresponding to `mean` in the computation formula. It has the same data type as input `rstd`, and only FLOAT32 is supported. Its shape is equal to the shape of `rstd`, which is [A1,...,Ai,1,...,1], where there are j ones after Ai, consistent with the length of the axes to be normalized. Empty tensors are supported.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Denotes the weight tensor, corresponding to `weight` in the formula. Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_mask</td>
      <td>Optional attribute</td>
      <td><ul><li>Denotes the mask of the outputs, with a fixed length of 3. A value of true indicates that the output at the corresponding position is not empty.</li><li>The default value is {true, true, true}.</li></ul></td>
      <td>LISTBOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>Output</td>
      <td>Denotes the output gradient of the backward propagation. Whether it is output is controlled by element 0 of `output_mask`, corresponding to `gradInputOut` in the computation formula. It is output when element 0 of `output_mask` is true. It has the same data type as input x, and its shape is equal to the shape of `x`. Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>Output</td>
      <td>Denotes the gradient of the weight in the backward propagation. Whether it is output is controlled by element 1 of `output_mask`, corresponding to `gradWeightOut` in the computation formula. It is output when element 1 of `output_mask` is true. It has the same data type as input `gamma`, and its shape is equal to the shape of `pd_beta`. Empty tensors are supported.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>Output</td>
      <td>Denotes the gradient of the bias in the backward propagation. Whether it is output is controlled by element 2 of `output_mask`, corresponding to `gradBiasOut` in the computation formula. It is output when element 2 of `output_mask` is true. It has the same data type as input `gamma`. Its shape is equal to the shape of `pd_gamma`. Empty tensors are supported.</td>
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
| aclnn interface  | [test_aclnn_layer_norm_grad_v3](examples/test_aclnn_layer_norm_grad_v3.cpp) | Calls the LayerNormGradV3 operator through the [aclnnLayerNormBackward](docs/aclnnLayerNormBackward_en.md) interface. |
| Graph mode | [test_geir_layer_norm_grad_v3](examples/test_geir_layer_norm_grad_v3.cpp) | Calls the LayerNormGradV3 operator by constructing a graph through the [Ascend IR](docs/ascendirLayerNormGradV3.md). |
