# InstanceNormGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Backpropagation of Instance Normalization. Given the upstream gradient `dy` and the forward `x`, `variance`, `mean`, and `gamma`, the operator computes `pd_x`, `pd_gamma`, and `pd_beta`. `pd_x` is reduced over the spatial dimensions `(D, H, W)` for each `(N, C)` pair; `pd_gamma` and `pd_beta` are further reduced over `N` in addition to the spatial dimensions, keeping only the `C` dimension.
- Formula (`variance` is the original variance, `ε` is the compile-time constant `1e-6`, `rstd = (variance + ε)^(-1/2)`, `m = D*H*W`, `R = {D,H,W}`, and `x_hat = (x - mean) * rstd`):

  $$
  pd\_xl = dy \cdot gamma
  $$

  $$
  pd\_var = \sum_{R} \left( -0.5 \cdot pd\_xl \cdot (x - mean) \cdot (variance + \varepsilon)^{-3/2} \right)
  $$

  $$
  pd\_mean = \sum_{R} \left( -1.0 \cdot pd\_xl \cdot rstd \right)
  $$

  $$
  pd\_x = pd\_xl \cdot rstd + pd\_var \cdot \frac{2}{m} \cdot (x - mean) + pd\_mean \cdot \frac{1}{m}
  $$

  $$
  pd\_gamma = \sum_{N,R} (dy \cdot x\_hat), \quad pd\_beta = \sum_{N,R} (dy)
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
      <td>Gradient backpropagated from the upper layer, corresponding to `dy` in the formula. Its shape and data type are consistent with those of the input parameter `x`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input of the forward operator, corresponding to `x` in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Input</td>
      <td>Variance of each instance, corresponding to `variance` in the formula. Its data type is consistent with that of the input parameter `dy`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Mean of each instance, corresponding to `mean` in the formula. Its shape and data type are consistent with those of the input parameter `variance`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Scaling tensor in the standardization process, corresponding to `gamma` in the formula. Its data type is consistent with that of the input parameter `dy`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_x</td>
      <td>Output</td>
      <td>Gradient with respect to `x`, corresponding to `pd_x` in the formula. Its shape and data type are consistent with those of the input parameter `x`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_gamma</td>
      <td>Output</td>
      <td>Gradient with respect to `gamma`, corresponding to `pd_gamma` in the formula. Its shape and data type are consistent with those of the input parameter `gamma`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
    <tr>
      <td>pd_beta</td>
      <td>Output</td>
      <td>Gradient with respect to `beta`, corresponding to `pd_beta` in the formula. Its shape and data type are consistent with those of the input parameter `gamma`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDHWC</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation description: the implementation is deterministic.
- The shapes of the input parameters `dy` and `x` and the output `pd_x` must be exactly the same.
- The input parameters `variance` and `mean` must have the same shape, and the number of elements must be N×C (N is dimension 0 of `dy`, and C is the last dimension of `dy`).
- The number of elements of the input parameter `gamma` and of the outputs `pd_gamma` and `pd_beta` must all be C.
- The data types of all inputs and outputs must be identical, either all FLOAT16 or all FLOAT32.
- When `x` is an empty tensor, `pd_x` is also an empty tensor, and all elements of `pd_gamma` and `pd_beta` are 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_instance_norm_grad](examples/test_geir_instance_norm_grad.cpp) | Invokes the InstanceNormGrad operator through graph construction with the [operator IR](op_graph/instance_norm_grad_proto.h). |
