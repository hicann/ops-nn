# InplaceApplyAdamWithAmsgrad

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     x    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

InplaceApplyAdamWithAmsgrad performs a single-step in-place update on the four state tensors var, m, v, and vhat of the AMSGrad optimizer, with semantics aligned with the TensorFlow ResourceApplyAdamWithAmsgrad. var, m, v, and vhat are written back with the updated values in place. grad and the six scalar hyperparameters participate in the computation, and all tensor inputs and outputs share the same data type. This operator is invoked in GE graph mode.

The formulas are as follows. The update dependency order is fixed as m→v→vhat (using the new v)→var (using the new m):

$$
\alpha = lr \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
$$

$$
m = m + (grad - m) \times (1 - beta1)
$$

$$
v = v + (grad \times grad - v) \times (1 - beta2)
$$

$$
vhat = where(vhat < v,\ v,\ vhat)
$$

$$
var = var - \frac{m \times \alpha}{\sqrt{vhat} + epsilon}
$$

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
<td>The weight state to be updated, written back in place, corresponding to var in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>m</td>
<td>Input</td>
<td>The first moment state, written back in place, corresponding to m in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>v</td>
<td>Input</td>
<td>The second moment state, written back in place, corresponding to v in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>vhat</td>
<td>Input</td>
<td>The second moment running max state, written back in place, corresponding to vhat in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta1_power</td>
<td>Input</td>
<td>The scalar hyperparameter β1^t, corresponding to beta1_power in the formula, maintained by the framework side step by step.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta2_power</td>
<td>Input</td>
<td>The scalar hyperparameter β2^t, corresponding to beta2_power in the formula, maintained by the framework side step by step.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>The learning rate scalar hyperparameter, corresponding to lr in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta1</td>
<td>Input</td>
<td>The scalar hyperparameter β1, corresponding to beta1 in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta2</td>
<td>Input</td>
<td>The scalar hyperparameter β2, corresponding to beta2 in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>epsilon</td>
<td>Input</td>
<td>The scalar hyperparameter ε, corresponding to epsilon in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>The gradient tensor, corresponding to grad in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>A semantic placeholder attribute. The default value is false. The current implementation does not perform locking, and the numerical behavior is equivalent to use_locking=false.</td>
<td>Bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The five state tensors var, m, v, vhat, and grad must share the same data type (FLOAT, FLOAT16, or BFLOAT16). The six scalar inputs beta1_power, beta2_power, lr, beta1, beta2, and epsilon are fixed to FLOAT.
- The five state tensors var, m, v, vhat, and grad must have exactly the same shape, and broadcasting is not supported among the five.
- beta1_power, beta2_power, lr, beta1, beta2, and epsilon must be rank-0 scalars or tensors with shape [1].
- Only the ND format is supported.
- The rank of the state tensors must not exceed 8.

## Invocation Description

<table style="table-layout: fixed; width: 1000px">
<colgroup>
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 600px">
</colgroup>
<thead>
<tr>
<th>Invocation method</th>
<th>Sample code</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>Graph-mode invocation</td>
<td><a href="./examples/test_geir_inplace_apply_adam_with_amsgrad.cpp">test_geir_inplace_apply_adam_with_amsgrad</a></td>
<td>Invoke the InplaceApplyAdamWithAmsgrad operator by constructing a graph through the operator IR.</td>
</tr>
</tbody>
</table>
