# FusedMulApplyKerasMomentum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Fused operator of the Keras-style Momentum SGD optimizer. It first performs one fused gradient multiplication to obtain the gradient, then updates the weight var and the momentum accumulator accum in place according to the Momentum SGD formulas. Both standard Momentum and Nesterov Momentum modes are supported. The computation semantics align with the `tf.raw_ops.ResourceApplyKerasMomentum` API in TensorFlow.
- Formula:

  Gradient and momentum accumulator updates (identical in both modes):

  $$
  grad = x1 \times x2
  $$

  $$
  accum_{new} = accum \times momentum - grad \times lr
  $$

  - Standard mode (use_nesterov = false):

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov mode (use_nesterov = true):

  $$
  var_{new} = var + accum_{new} \times momentum - grad \times lr
  $$

  Here, `x1` and `x2` are gradient-related inputs, `lr` is the learning rate, `momentum` is the momentum coefficient, `accum` is the momentum accumulator, and `var` is the weight parameter to be updated.
- Accuracy description: A FLOAT16 input is first promoted to FLOAT32 inside the Kernel, accumulated with high-precision FLOAT32 throughout, and finally converted back to FLOAT16 for output.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 150px">
<col style="width: 180px">
<col style="width: 400px">
<col style="width: 180px">
<col style="width: 120px">
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
    <td>Weight parameter to be updated, which corresponds to var in the formula. It has the same name as the output port var (in-place alias). The Kernel writes the output back to this input storage.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input</td>
    <td>Momentum accumulator, which corresponds to accum in the formula. It has the same name as the output port accum (in-place alias). The Kernel writes the output back to this input storage. The shape and dtype must be the same as those of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, which corresponds to lr in the formula. A scalar tensor with shape [1], and the dtype is the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1</td>
    <td>Input</td>
    <td>Gradient-related input, which corresponds to x1 in the formula. The shape and dtype must be the same as those of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>Input</td>
    <td>Momentum coefficient, which corresponds to momentum in the formula. A scalar tensor with shape [1], and the dtype is the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Gradient-related input, which corresponds to x2 in the formula. A scalar tensor with shape [1], and the dtype is the same as that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>Output</td>
    <td>Updated weight parameter, which corresponds to var_new in the formula. It shares device memory with the input var (in-place alias), and its shape and dtype are the same as those of the input var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Output</td>
    <td>Updated momentum accumulator, which corresponds to accum_new in the formula. It shares device memory with the input accum (in-place alias), and its shape and dtype are the same as those of the input accum.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Optional attribute</td>
    <td><ul><li>Whether to lock and protect the update operation.</li><li>Default value: false.</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>Optional attribute</td>
    <td><ul><li>Whether to use Nesterov momentum. true indicates Nesterov mode, and false indicates standard mode.</li><li>Default value: false.</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- var, accum, and x1 must have the same data type and data format, and their shapes must be identical.
- The output ports of var and accum have the same names as their input ports (in-place aliases). Their shapes are the same as those of the corresponding inputs. You must treat these two inputs as writable.
- lr, momentum, and x2 are scalar tensors with shape [1], and their data type is the same as that of var.
- The number of dimensions of var, accum, and x1 does not exceed 8.
- Non-contiguous tensor inputs are not supported.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_fused_mul_apply_keras_momentum.cpp">Sample</a></td>
    <td>The production path of this operator is the GE IR graph mode. Call the FusedMulApplyKerasMomentum operator through graph construction with the <a href="./op_graph/fused_mul_apply_keras_momentum_proto.h">operator IR</a>. TensorFlow training generates this operator through tf_plugin and fused PASS scheduling.</td>
  </tr>
</tbody></table>
