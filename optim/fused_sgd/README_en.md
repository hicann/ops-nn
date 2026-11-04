# FusedSgd

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- **Operator function**: Implements the fusedSgd operator. It fuses multiple fine-grained operations that are originally scattered in the traditional SGD update process (such as gradient scaling, weight decay, momentum update, and parameter assignment) into a single NPU kernel for execution.
- **Formula**:

  $$
  \begin{aligned}
  &\tilde{g}_t = \begin{cases}
  g_t / s & s \neq \text{None} \\
  g_t & \text{otherwise}
  \end{cases} \\

  &\hat{g}_t = \begin{cases}
  -\tilde{g}_t & \text{maximize} \\
  \tilde{g}_t & \text{otherwise}
  \end{cases} \\

  &\bar{g}_t = \hat{g}_t + weightDecay \cdot \theta_t \\

  &v_{t+1} = \begin{cases}
  \bar{g}_t & \text{first step} \\
  \mu v_t + (1-dampening)\bar{g}_t & \text{otherwise}
  \end{cases} \\

  &g_t^{\text{final}} = \begin{cases}
  \bar{g}_t + \mu v_{t+1} & \text{nesterov} \\
  v_{t+1} & v_t \neq \text{None} \\
  \bar{g}_t & \text{otherwise}
  \end{cases} \\

  &\theta_{t+1} = \theta_t - lr \cdot g_t^{\text{final}} \\
  &g_{t+1} = \tilde{g}_t

  \end{aligned}
  $$

## Parameters

<table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
<col style="width: 155px">
<col style="width: 162px">
<col style="width: 380px">
<col style="width: 276px">
<col style="width: 107px">
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
    <td>params</td>
    <td>Input/Output</td>
    <td>Parameters to be updated, corresponding to θ in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grads</td>
    <td>Input/Output</td>
    <td>Gradients corresponding to the parameters to be updated, corresponding to g in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Momentum corresponding to the parameters to be updated, corresponding to v in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradScale</td>
    <td>Input</td>
    <td>Gradient scaling size, corresponding to s in the formula.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Momentum corresponding to the parameters to be updated, corresponding to v in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weightDecay</td>
    <td>Attribute</td>
    <td>Weight decay value, corresponding to weightDecay in the formula. Default: 0.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Attribute</td>
    <td>Momentum value, corresponding to μ in the formula. Default: 0.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Attribute</td>
    <td>Learning rate, corresponding to lr in the formula. Default: 1e-3.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dampening</td>
    <td>Attribute</td>
    <td>Momentum dampening coefficient, corresponding to dampening in the formula. Default: 0.</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>nesterov</td>
    <td>Attribute</td>
    <td>Specifies whether to enable Nesterov momentum, corresponding to nesterov in the formula. Default: False.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>maximize</td>
    <td>Attribute</td>
    <td>Specifies whether to maximize the objective function, corresponding to maximize in the formula. Default: False.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>isFirstStep</td>
    <td>Attribute</td>
    <td>Specifies whether this is the first-step update, corresponding to FirstStep in the formula. Default: True.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The data types of params, grads, x, and gradScale are within the supported range.
- params, grads, and x, and each tensor in them, have the same data type.
- The shape dimensions of the tensors in params, grads, and x are less than or equal to 8, and the shape of gradScale is [1].
- The values of lr, momentum, weightDecay, and dampening are greater than or equal to 0.
- The tensors at the same index in params, grads, and x (when x is not empty) have the same shape.
- params, grads, and x (when x is not empty) cannot contain null pointers.

## Invocation Description

| Invocation method  | Sample code                                                     | Description                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn invocation | [test_aclnn_fused_sgd](./examples/test_aclnn_fused_sgd.cpp) | Invokes the FusedSgd operator through the [aclnnFusedSgd](docs/aclnnFusedSgd_en.md) interface. |
