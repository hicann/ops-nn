# ApplyKerasMomentum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- **Operator function**: Perform a single-step parameter update of the Keras Momentum optimizer. The operator updates the momentum accumulator accum based on the momentum coefficient momentum, learning rate lr, and gradient grad, and updates the weight parameter var in place in standard mode or Nesterov mode (inplace semantics). The computation semantics are aligned with the `tf.raw_ops.ResourceApplyKerasMomentum` API in TensorFlow.

- **Formula**:

  $$
  \begin{aligned}
  accum_{new} &= momentum \cdot accum - lr \cdot grad
  \end{aligned}
  $$

  - Standard mode (use_nesterov = 0):

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov mode (use_nesterov = 1):

  $$
  var_{new} = var + momentum \cdot accum_{new} - lr \cdot grad
  $$

  Here `momentum` is the momentum coefficient, `lr` is the learning rate, `grad` is the current gradient, `accum` is the momentum accumulator, and `var` is the weight parameter to be updated.

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
    <td>Input / Output (inplace)</td>
    <td>Weight parameter to be updated, corresponding to var in the formula. The kernel updates it in place, and the GE IR output var_out shares device memory with input var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input (inplace update)</td>
    <td>Momentum accumulator, corresponding to accum in the formula. The shape/dtype must be the same as those of var. The kernel explicitly writes back to the input GM address.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. A 1-element scalar tensor with shape={1}, and its dtype is FLOAT.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient tensor, corresponding to grad in the formula. The shape/dtype must be the same as those of var.</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient, corresponding to momentum in the formula. A 1-element scalar tensor with shape={1}, and its dtype is FLOAT.</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated var tensor, which shares device memory with input var (inplace).</td>
    <td>FLOAT16, FLOAT, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Indicates whether to lock the update operation. An attribute of the Bool type. Default value: false.</td>
    <td>Bool</td>
    <td>—</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>Attribute</td>
    <td>Indicates whether to use Nesterov momentum, corresponding to use_nesterov in the formula. An attribute of the Bool type. Default value: false (false = standard mode, true = Nesterov mode).</td>
    <td>Bool</td>
    <td>—</td>
  </tr>
</tbody></table>

> **Note**: accum is updated in place (consistent with TensorFlow ResourceApplyKerasMomentum). It is not returned as an explicit output, and the computation result is directly written back to the input address.

## Constraints

- var, accum, and grad must have the same data type and shape.
- lr and momentum are scalar tensors (shape={1}), and their data type is FLOAT.
- use_locking and use_nesterov are attributes of the Bool type, with a default value of false.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_apply_keras_momentum](./examples/arch35/test_geir_apply_keras_momentum.cpp) | Invoke the ApplyKerasMomentum operator through graph construction using the [operator IR](./op_graph/apply_keras_momentum_proto.h). |
