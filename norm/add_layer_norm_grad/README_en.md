# AddLayerNormGrad

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

## Function

- Operator function: LayerNorm is a normalization method that normalizes the input data of a network layer into a distribution with a mean of 0 and a variance of 1. The LayerNormGrad operator is a key operator in the backpropagation stage of deep learning, and it is mainly used to compute the gradients of the LayerNorm operation. The AddLayerNormGrad operator fuses Add and LayerNormGrad to reduce data transfer in and out.

- Formula:

  - Forward formula (D is the size of the reduce axis):

    $$
    x= inputx1 + inputx2
    $$

    $$
    \operatorname{LayerNorm}(x_i)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*\gamma + \beta
    $$

    $$
    其中\operatorname{E}(x)=\frac{1}{D}\sum_{i=1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x)=\frac{1}{D}\sum_{i=1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - Backward formula:

    $$
    x= inputx1 + inputx2
    $$

    $$
    dxOut = \sum_{i}{inputdy_i * \gamma_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}} + dsumOptional
    $$

    $$
    dgammaOut = \sum_{j}{inputdy_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}}
    $$

    $$
    dbetaOut = \sum_{j}{inputdy_i}
    $$

    Where:
    - $\hat{x_j}$:

      $$
      \hat{x_j}=({x_i-\operatorname{E}(x)}) * {rstd}
      $$

    - $rstd$:

      $$
      rstd=\frac {1}{\sqrt{\operatorname{Var}(x)}}
      $$

    - $\frac{{\rm d}\hat{x_j}}{{\rm d}x_i}$:

      $$
      \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}  x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x)}}-\frac{1}{\operatorname{Var}(x)}  (x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x)}{\rm dx}
      $$

      Here $\delta_{ij}$=1 when i=j, and $\delta_{ij}$=0 when i!=j.

    - $\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}$:

      $$
      \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
      $$

      Here D is the number of elements of x that participate in the mean computation.

    - $\frac{\rm d \operatorname{Var}(x)}{\rm dx}$:

      $$
      \frac{\rm d \operatorname{Var}(x)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var}(x)}}(x_i-\operatorname{E}(x))
      $$

    - Simplified $dxOut$:

      $$
      dxOut = rstd * ({inputdy_i * \gamma_j} - \frac{1}{D} * (\sum_{j}{inputdy_i * \gamma_j} + \hat {x_j} * \sum_{j}{inputdy_i * \gamma_j * \hat{x_j}})) + dsumOptional
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
      <td>The main grad input, corresponding to `inputdy` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>The input x1 of the forward fused operator, corresponding to `inputx1` in the formula. The shape and data type are the same as the shape and data type of `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The input x2 of the forward fused operator, corresponding to `inputx2` in the formula. The shape and data type are the same as the shape and data type of `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Represents the reciprocal of the standard deviation of the sum of the forward inputs x1 and x2, corresponding to `rstd` in the formula. The shape must satisfy the broadcast relationship with `dy` (the leading dimensions are the same as the leading dimensions of `dy`, and the leading dimensions refer to the dimensions of `dy` minus the dimensions of `gamma`, indicating the dimensions that do not need normalization).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Represents the mean of the sum of the forward inputs x1 and x2, corresponding to `E(x)` in the formula. The shape must satisfy the broadcast relationship with `dy` (the leading dimensions are the same as the leading dimensions of `dy`, and the leading dimensions refer to the dimensions of `dy` minus the dimensions of `gamma`, indicating the dimensions that do not need normalization).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Represents the dimensions that need normalization, corresponding to `γ` in the formula. The shape dimensions are the same as the trailing dimensions of `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dsum</td>
      <td>Optional input</td>
      <td>An additional backward gradient accumulation input, corresponding to the input `dsumOptional` in the formula. The shape and data type are the same as the shape and data type of `dy`.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>
        <ul>
          <li>Represents the gradient of the sum of the inputs x1 + x2, corresponding to `dxOut` in the formula. The shape and data type are the same as the shape and data type of `dy`.</li>
          <li>The data type and shape are the same as the input x1.</li>
        </ul>
      </td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>Output</td>
      <td>
        <ul>
          <li>Represents the gradient of the input parameter gamma, corresponding to `dgammaOut` in the formula. The shape is the same as the input `gamma`.</li>
          <li>The data type is fixed to FLOAT32, and the shape is the same as the input gamma.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>Output</td>
      <td>
        <ul>
          <li>Represents the backward gradient of the forward input parameter beta, corresponding to `dbetaOut` in the formula. The shape is the same as the input `gamma`.</li>
          <li>The data type is fixed to FLOAT32, and the shape is the same as the input gamma.</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas inference products</term> and <term>Atlas training products</term>: the BFLOAT16 data type is not supported for the input parameters `dy`, `x1`, and `x2` and the output parameters `dsum` and `dx`.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_add_layer_norm_grad](examples/test_aclnn_add_layer_norm_grad.cpp) | Invoke the AddLayerNormGrad operator through the [aclnnAddLayerNormGrad](docs/aclnnAddLayerNormGrad_en.md) interface. |
| Graph mode | -  | Invoke the AddLayerNormGrad operator by constructing a graph through the [operator IR](op_graph/add_layer_norm_grad_proto.h).         |
