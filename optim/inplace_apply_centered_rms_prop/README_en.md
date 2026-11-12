# InplaceApplyCenteredRMSProp

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Implements parameter updates of the Centered RMSProp optimizer. During model training, updates the model parameter `var`, the first moment of the gradient `mg`, the second moment of the gradient `ms`, and the momentum `mom` in place based on the current gradient, learning rate, and decay coefficient.
- Formula:

  $$
  \begin{aligned}
  mg_t &= \rho \cdot mg_{t-1} + (1 - \rho) \cdot g_t \\
  ms_t &= \rho \cdot ms_{t-1} + (1 - \rho) \cdot g_t^2 \\
  mom_t &= \mu \cdot mom_{t-1} + lr \cdot \frac{g_t}{\sqrt{ms_t - mg_t^2 + \epsilon}} \\
  var_t &= var_{t-1} - mom_t
  \end{aligned}
  $$

  Here, $var$ denotes the model parameters, $mg$ denotes the first moment of the gradient, $ms$ denotes the second moment of the gradient, $mom$ denotes the momentum, $g_t$ denotes the gradient of the current step, $lr$ denotes the learning rate, $\rho$ denotes the decay coefficient, $\mu$ denotes the momentum coefficient, and $\epsilon$ denotes the numerical stability constant. $\epsilon$ is inside the square root, which is consistent with the computation method of TensorFlow `ResourceApplyCenteredRMSProp`.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
    <td>var</td>
    <td>Input</td>
    <td>Model parameters to be updated. The shape is the same as those of mg, ms, mom, and grad.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>Input</td>
    <td>Exponential moving average of the first moment of the gradient. The shape is the same as that of var.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>Input</td>
    <td>Exponential moving average of the second moment of the gradient. The shape is the same as that of var.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>Input</td>
    <td>Momentum. The shape is the same as that of var.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate. Must be a scalar or a 1D tensor containing only one element.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>Input</td>
    <td>Decay coefficient. Must be a scalar or a 1D tensor containing only one element.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient. Must be a scalar or a 1D tensor containing only one element.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Numerical stability constant. Must be a scalar or a 1D tensor containing only one element.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Gradient of the current step. The shape is the same as that of var.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Optional compatibility attribute. Default value: false. The value does not affect the computation result.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated model parameters, which share storage with the input var.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>Output</td>
    <td>Updated first moment of the gradient, which shares storage with the input mg.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>Output</td>
    <td>Updated second moment of the gradient, which shares storage with the input ms.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>Output</td>
    <td>Updated momentum, which shares storage with the input mom.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The shapes of var, mg, ms, mom, and grad must be completely identical.
- lr, rho, momentum, and epsilon must be scalars or 1D tensors containing only one element.
- The data types of all inputs and outputs must be the same. FLOAT and FLOAT16 are supported.
- var, mg, ms, and mom are all updated in place, and the outputs share storage with the corresponding inputs.
- <term>Ascend 950PR&950DT products</term>:
  - var, mg, ms, mom, grad, and their corresponding outputs support only the ND format, and the rank does not exceed 8.
  - Empty tensors are supported.
  - A FLOAT16 input is computed internally with FLOAT precision, and the result is converted to FLOAT16 for output.
- `use_locking` is used only for API compatibility. The current implementation does not provide an additional mutex lock.
- This operator supports only invocation in the GE graph mode and does not provide a public aclnn API.

## Invocation Description

| Invocation method | Sample code | Description |
|:---------|:---------|:-----|
| GE graph mode | [test_geir_inplace_apply_centered_rms_prop](./examples/arch35/test_geir_inplace_apply_centered_rms_prop.cpp) | Calls the InplaceApplyCenteredRMSProp operator through graph construction with the [operator IR](./op_graph/inplace_apply_centered_rms_prop_proto.h). |
