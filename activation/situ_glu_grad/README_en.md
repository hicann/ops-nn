# SituGluGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×    |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×    |

## Function

- Operator function: Backward gradient computation of the SiTU Gated Linear Unit activation function. Based on the upstream gradient grad_y and the forward input x, it computes the gradient grad_x of input x.

- Formula or computation process:

  For the given input tensor x (with dimensions [..., cut], where cut is an even number; let h = cut // 2) and the upstream gradient grad_y (with dimensions [..., h]):

  1. Split x into gate and up according to activate_left (each with shape [..., h]).

  2. Compute the forward intermediate quantities:

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     When linear_beta > 0: $up' = linear\_beta \cdot \tanh(up / linear\_beta)$; otherwise $up' = up$.

  3. Product rule:

     $$
     grad\_situ\_a = grad\_y \cdot up', \quad grad\_up' = grad\_y \cdot situ\_a
     $$

  4. Gate gradient:

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right]
     $$

  5. Up gradient:

     - When linear_beta > 0: $grad\_up = grad\_up' \cdot (1 - \tanh^2(up / linear\_beta))$
     - When linear_beta ≤ 0: $grad\_up = grad\_up'$

  6. Concatenate grad_gate and grad_up into grad_x according to activate_left (with the same shape as x).

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>grad_y</td>
      <td>Input</td>
      <td>Upstream gradient of backpropagation. The dim dimension size is half of that of x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Forward input. The dim dimension is evenly split into the gate part and the up part. The data type must be the same as grad_y.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>Attribute</td>
      <td><ul><li>Split dimension, which must be the same as the forward SituGlu.</li><li>Value range: [-x.dim(), x.dim()-1].</li><li>Default value: -1.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the SiTU gate part, which must be the same as the forward SituGlu.</li><li>Default value: 1.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>Attribute</td>
      <td><ul><li>Scaling coefficient of the linear tanh in the up path, which must be the same as the forward SituGlu.</li><li>Default value: 0.0.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>Attribute</td>
      <td><ul><li>Whether gate is the first half when x is split, which must be the same as the forward SituGlu.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>Output</td>
      <td>Gradient of input x, with exactly the same shape as x.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Non-contiguous tensors are supported.
- beta, linear_beta, activate_left, and dim must have the same values as those in the forward SituGlu invocation; otherwise the gradient result is incorrect.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | - | Calls the SituGluGrad operator through the [aclnnSituGluGrad](./docs/aclnnSituGluGrad_en.md) interface. |
| GE graph mode | -  | Calls the SituGluGrad operator by building a graph with the [operator IR](./op_graph/situ_glu_grad_proto.h).         |
| PyTorch API |  -  | Calls the SituGluGrad operator through the [situ_glu_grad](./docs/torchapi_situ_glu_grad_en.md) interface.         |
