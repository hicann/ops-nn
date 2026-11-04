# FusedBiasLeakyReluGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: this is a three-in-one backward gradient operator of BiasAdd + LeakyReLU + Scale, derived from the backward pass of FusedBiasLeakyReLU in the MMCV framework. Given the upstream gradient y_grad and the forward feature values features, it generates a gradient mask according to the sign of features, multiplies it element-wise by y_grad, and then multiplies the result by the scale factor to output the gradient x_grad of the input features.

- Formula:

  $$
  x\_grad[i] = scale \cdot y\_grad[i] \cdot \begin{cases} 1.0 & \text{if } features[i] > 0 \\ negative\_slope & \text{if } features[i] \leq 0 \end{cases}
  $$

## Parameters

<table style="table-layout: fixed; width: 100%"><colgroup>
<col style="width: 15%">
<col style="width: 12%">
<col style="width: 35%">
<col style="width: 20%">
<col style="width: 18%">
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
    <td>y_grad</td>
    <td>Input</td>
    <td>Gradient tensor passed back from upstream; its data type must be consistent with features.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>features</td>
    <td>Input</td>
    <td>Forward feature value tensor, used for the sign check (features > 0 serves as the gradient mask). Its shape must be broadcast-compatible with y_grad.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>negative_slope</td>
    <td>Attribute</td>
    <td>Slope coefficient of the negative half axis of LeakyReLU. The default value is 0.2.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Attribute</td>
    <td>Variance-preserving scale factor. The default value is sqrt(2) ≈ 1.414213562373.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>x_grad</td>
    <td>Output</td>
    <td>Gradient of the input features, with the shape of the broadcast result of y_grad and features.</td>
    <td>Same as y_grad</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The data types of y_grad and features must be the same.
- The shapes of y_grad and features must satisfy the broadcast rules, and the rank after broadcasting does not exceed 8.
- Empty tensors are supported (an empty output is returned when the number of elements is 0).

## Invocation Description

| Invocation method | Sample code | Description |
|---|---|---|
| Graph mode | [test_geir_fused_bias_leaky_relu_grad](./examples/test_geir_fused_bias_leaky_relu_grad.cpp) | Invokes the FusedBiasLeakyReluGrad operator by constructing the graph through the [operator IR](./op_graph/fused_bias_leaky_relu_grad_proto.h). |
