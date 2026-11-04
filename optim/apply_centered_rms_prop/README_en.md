# ApplyCenteredRMSProp

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |
| Atlas 200I/500 A2 inference products | √ |
| Atlas inference products | √ |
| Atlas training products | √ |

## Function

- Operator function: Implements the RMSProp optimizer update step with the "centered" correction, used for model parameter updates in the deep learning training phase. On top of standard RMSProp, the operator additionally maintains the exponential moving average (mg) of the first moment of the gradient and uses `ms - mg²` as the variance estimate instead of the original second moment `ms`, providing a more stable adaptive learning rate.
- Formula:

  **Step 1 — First moment update**:

  $$
  mg_t = \rho \cdot mg_{t-1} + (1 - \rho) \cdot g_t
  $$

  **Step 2 — Second moment update**:

  $$
  ms_t = \rho \cdot ms_{t-1} + (1 - \rho) \cdot g_t^2
  $$

  **Step 3 — Variance estimation**:

  $$
  \text{variance}_t = ms_t - mg_t^2
  $$

  **Step 4 — Denominator computation (epsilon is added inside sqrt)**:

  $$
  \text{denom}_t = \sqrt{\text{variance}_t + \epsilon}
  $$

  **Step 5 — Momentum update (when momentum > 0)**:

  $$
  mom_t = \mu \cdot mom_{t-1} + lr \cdot \frac{g_t}{\text{denom}_t}
  $$

  $$
  var_t = var_{t-1} - mom_t
  $$

  **Step 6 — Direct update (when momentum == 0)**:

  $$
  var_t = var_{t-1} - lr \cdot \frac{g_t}{\text{denom}_t}
  $$

  Where:
  - `var`: model parameters to be updated
  - `mg`: exponential moving average of the first moment of the gradient
  - `ms`: exponential moving average of the second moment of the gradient
  - `mom`: momentum buffer
  - `g_t`: gradient of the current step
  - `lr`: learning rate
  - `ρ` (rho): decay coefficient
  - `μ` (momentum): momentum coefficient
  - `ε` (epsilon): numerical stability constant

> **Note**: epsilon is added inside sqrt (`√(x + ε)`), which is consistent with the semantics of TensorFlow `tf.raw_ops.ApplyCenteredRMSProp`. There is a slight numerical difference from the `√x + ε` of PyTorch.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 300px">
<col style="width: 250px">
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
    <td>var</td>
    <td>Input</td>
    <td>Model parameters to be updated. Empty tensors are supported. The shape is consistent with those of mg/ms/mom/grad.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>Input</td>
    <td>Exponential moving average of the first moment of the gradient. Empty tensors are supported. The shape is consistent with that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>Input</td>
    <td>Exponential moving average of the second moment of the gradient. Empty tensors are supported. The shape is consistent with that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>Input</td>
    <td>Momentum buffer. Empty tensors are supported. The shape is consistent with that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate. Must be a 0-D scalar tensor.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>Input</td>
    <td>Decay coefficient. Must be a 0-D scalar tensor.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient. Must be a 0-D scalar tensor.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Numerical stability constant. Must be a 0-D scalar tensor.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Gradient of the current step. Empty tensors are supported. The shape is consistent with that of var.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Optional attribute that specifies whether to lock the update. Default value: false.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated model parameters, which share storage with the input var (in place). mg/ms/mom are also updated in place on the inputs, which is consistent with the canndev IR and is not declared as separate outputs.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The shapes of the five ND tensors var, mg, ms, mom, and grad must be exactly the same.
- lr, rho, momentum, and epsilon must be 0-D scalar tensors.
- The data types of all inputs and outputs must be the same (FLOAT, FLOAT16, or BFLOAT16).
- The operator declares only one output, var, which shares storage with the input var; mg/ms/mom are updated in place on the inputs.
- Sparse gradients are not supported.
- For FLOAT16/BFLOAT16 inputs, the internal computation uses the FP32 mixed-precision path and finally casts back to the original dtype.
- fp16 overflow boundary: when grad values approach the fp16 maximum 65504, the internal FP32 computation path prevents intermediate overflow, but if the result exceeds the range when finally cast back to fp16, it is saturated and truncated.
- The ACLNN interface executes `AutoContiguous()` to make non-contiguous inputs contiguous; the GE graph path uses the ND format according to the registration constraints.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_apply_centered_rms_prop](./examples/test_geir_apply_centered_rms_prop.cpp) | Builds the graph and invokes the ApplyCenteredRMSProp operator through the [operator IR](./op_graph/apply_centered_rms_prop_proto.h). |
