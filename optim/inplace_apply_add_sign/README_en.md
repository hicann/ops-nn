# InplaceApplyAddSign

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

InplaceApplyAddSign is an equivalent implementation of TensorFlow `ResourceApplyAddSign` on Ascend and belongs to the training optimizer operator family. The operator updates the trainable variable var and the first momentum m in place using the AddSign algorithm. In each training step, the operator computes an adaptive step size based on sign consistency with the current gradient grad, updates var and m, and writes the updated values back in place. The operator applies to training scenarios that use the AddSign optimizer, such as CV, NLP, recommendation systems, and generative models.

Formula:

$$
m_{out} = beta \times m + (1 - beta) \times grad
$$

$$
var_{out} = var - lr \times (alpha + sign\_decay \times sign(grad) \times sign(m_{out})) \times grad
$$

Here `sign(x)` follows the numpy.sign semantics: +1 when x>0, -1 when x<0, 0 when x==0, and 0 for NaN. var and m are written back in place. lr, alpha, sign_decay, and beta are rank=0 scalar inputs and participate in the computation according to the numpy broadcasting rules.

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
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
</tr>
</thead>
<tbody>
<tr>
<td>var</td>
<td>Input</td>
<td>Trainable variable to be updated, corresponding to var in the formula. The updated value is written back in place.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>m</td>
<td>Input</td>
<td>First momentum buffer, corresponding to m in the formula. The updated value is written back in place.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>Learning rate, corresponding to lr in the formula. A rank=0 scalar.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>alpha</td>
<td>Input</td>
<td>Baseline increment of the step size, corresponding to alpha in the formula. A rank=0 scalar.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>sign_decay</td>
<td>Input</td>
<td>Decay weight of sign consistency, corresponding to sign_decay in the formula. A rank=0 scalar.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>beta</td>
<td>Input</td>
<td>Momentum decay coefficient, corresponding to beta in the formula. A rank=0 scalar.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>Gradient of the current step, corresponding to grad in the formula.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>Indicates whether to lock the update of var and m. Default value: false. In Ascend single-operator execution mode, true and false behave in the same way.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- In Ascend single-operator execution mode, the use_locking attribute behaves in the same way for true and false. It is used only to align the API with TensorFlow.
- The three tensor inputs var, m, and grad must have the same shape and data type.
- lr, alpha, sign_decay, and beta must be scalars: both rank=0 scalars and scalar tensors with shape (1,) are accepted.
- The dimension range of tensor inputs is 0D to 8D.

## Invocation Description

| Invocation method | Invocation sample | Description |
|----------|----------|------|
| Graph-mode invocation | [test_geir_inplace_apply_add_sign](./examples/arch35/test_geir_inplace_apply_add_sign.cpp) | Invoke the InplaceApplyAddSign operator through graph construction using the [operator IR](./op_graph/inplace_apply_add_sign_proto.h). Static shape scenarios are covered. |
