# InplaceApplyPowerSign

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- **Operator function**: performs a single-step parameter update of the PowerSign optimizer. PowerSign decouples the update mechanism into two dimensions, "direction" and "magnitude", uses the sign consistency between the gradient and the first moment to adaptively adjust the update step size, and updates the weight `var` and the first moment `m` **in place** (inplace semantics). It corresponds to the TensorFlow `tf.raw_ops.ApplyPowerSign` / `tf.raw_ops.ResourceApplyPowerSign` interfaces.

- **Formula**:

  $$
  \begin{aligned}
  m_{t} &= \beta \cdot m_{t-1} + (1 - \beta) \cdot grad \\
  sign\_gm &= \text{sign}(m_{t}) \cdot \text{sign}(grad) \\
  var_{t} &= var_{t-1} - lr \cdot \exp(logbase \cdot sign\_decay \cdot sign\_gm) \cdot grad
  \end{aligned}
  $$

  Here, `sign(x)` is the sign function (it returns 1 when x > 0, -1 when x < 0, and 0 when x = 0), and `exp()` is the natural exponential function.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 600px">
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
    <td>Weight parameter tensor to be updated, corresponding to var in the formula. It shares the GM address with the graph output port var (inplace update).</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>Input</td>
    <td>First moment estimate tensor, corresponding to m in the formula. The shape/dtype must be consistent with var; it shares the GM address with the graph output port m (inplace update).</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. The shape must be a scalar tensor of [1].</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>logbase</td>
    <td>Input</td>
    <td>Logarithm base, corresponding to logbase in the formula. The shape must be a scalar tensor of [1].</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sign_decay</td>
    <td>Input</td>
    <td>Sign decay factor, corresponding to sign_decay in the formula. The shape must be a scalar tensor of [1].</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>Input</td>
    <td>First moment decay rate, corresponding to β in the formula, with the value range [0, 1). The shape must be a scalar tensor of [1].</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient tensor, corresponding to grad in the formula. The shape/dtype must be consistent with var.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Whether to lock during the update. The default value is false. The current implementation does not enforce a mutex lock; it serves only as a semantic placeholder.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated weight tensor, which shares the Device memory with input var (inplace). The shape/dtype is exactly the same as that of input var.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>Output</td>
    <td>Updated first moment estimate tensor, which shares the Device memory with input m (inplace). The shape/dtype is exactly the same as that of input m.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

- The shapes of the three tensors var, m, and grad must be consistent.
- lr, logbase, sign_decay, and beta must be scalar tensors with shape=[1].
- The DataTypes of all inputs and outputs must be consistent.
- FP16/BF16 inputs are internally promoted to FP32 for computation and cast back to the original type.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_inplace_apply_power_sign](./examples/arch35/test_geir_inplace_apply_power_sign.cpp) | Invokes the InplaceApplyPowerSign operator through graph construction with the [operator IR](./op_graph/inplace_apply_power_sign_proto.h). |
