# InplaceApplyFtrlV2

English | [简体中文](./README.md)

<!-- codespell:ignore FTRL Ftrl -->

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

InplaceApplyFtrlV2 is a single-step parameter update operator for the FTRL-Proximal (Follow The Regularized Leader - Proximal) online learning optimization algorithm. It is used in the large-scale sparse feature training phase of recommendation systems and CTR estimation. Based on the gradient `grad` of the current batch and a set of hyperparameters, the operator updates in place three state tensors that persist across calls: the model parameter `var`, the squared gradient accumulation `accum`, and the linear term `linear`. It also explicitly outputs the three updated tensors. Through L1 regularization, weights with `|linear| ≤ l1` are directly set to zero, producing a sparse solution.

Let `gs = grad + 2.0 * l2_shrinkage * var`, `accum_new = accum + grad * grad`, and `Δpow = accum_new^(-lr_power) - accum^(-lr_power)`. The update formulas are:

$$
linear_{new} = linear + gs - \frac{\Delta pow \cdot var}{lr}
$$

$$
var_{new} = \begin{cases} \frac{l1 \cdot sign(linear_{new}) - linear_{new}}{accum_{new}^{(-lr\_power)}/lr + 2.0 \cdot l2}, & |linear_{new}| > l1 \\ 0, & |linear_{new}| \le l1 \end{cases}
$$

When `l2_shrinkage=0`, `gs=grad`, and the behavior degenerates to FTRL V1. `out_accum = accum_new` (using the original `grad`, not `gs`).

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
<td>Model parameter to be updated, corresponding to var in the formula. It has the same name as the output var. The output GM address is aliased to the input GM address (inplace_with aliasing, a formal interface contract), and the input is updated in place.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>Input</td>
<td>Squared gradient accumulation (FTRL adaptive learning rate state, persisting across calls), corresponding to accum in the formula. Valid domain: accum >= 0.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>Input</td>
<td>Linear term (FTRL state variable, persisting across calls), corresponding to linear in the formula.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>Gradient of the current batch (read-only), corresponding to grad in the formula.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>Learning rate (scalar), corresponding to lr in the formula. It is broadcast to var.shape according to the numpy rules.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>Input</td>
<td>L1 regularization coefficient (scalar), which controls sparsity. It corresponds to l1 in the formula.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>Input</td>
<td>L2 regularization coefficient (scalar), which affects the denominator of the var_new formula. It corresponds to l2 in the formula.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2_shrinkage</td>
<td>Input</td>
<td>L2 shrinkage coefficient (scalar), corresponding to l2_shrinkage in the formula. When it is 0, the behavior degenerates to V1.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>lr_power</td>
<td>Input</td>
<td>Power of lr (scalar, usually -0.5), corresponding to lr_power in the formula. Valid domain: lr_power <= 0.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>Output</td>
<td>Updated weights, corresponding to var in the formula. It has the same name as the input var. This output's GM address is aliased to the GM address of the input var, and the in-place update takes effect (inplace_with aliasing, a formal interface contract).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>Output</td>
<td>Updated accumulation, corresponding to out_accum in the formula. It has the same name as the input accum. This output's GM address is aliased to the GM address of the input accum, and the in-place update takes effect (inplace_with aliasing, a formal interface contract).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>Output</td>
<td>Updated linear term, corresponding to linear in the formula. It has the same name as the input linear. This output's GM address is aliased to the GM address of the input linear, and the in-place update takes effect (inplace_with aliasing, a formal interface contract).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>Whether to use a mutex lock. The default value is false. The NPU does not have Ref semantics, so only false is supported.</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- Only the ND data format is supported.
- Automatic contiguous conversion is enabled for both inputs and outputs in the operator definition. Non-contiguous tensors are converted to contiguous tensors by the framework before execution, and the conversion may introduce additional data copies and performance overhead.
- Data type consistency: The data types of all 9 inputs (var/accum/linear/grad + lr/l1/l2/l2_shrinkage/lr_power) and the 3 outputs must be exactly the same (subject to var; mixed precision is not supported). FLOAT64 and integer types are not supported.
- Shape constraint: The shapes of the 4 tensors var/accumulation/linear/grad must be exactly the same, and broadcast between tensors is not supported. The rank range of the 4 tensors is [0, 8].
- Scalar constraint: The 5 scalar inputs lr/l1/l2/l2_shrinkage/lr_power are rank-0 or shape [1], and are broadcast to var.shape according to the numpy rules.
- Scalar value range: lr > 0, l1 >= 0, l2 >= 0, l2_shrinkage >= 0, accum >= 0, and lr_power <= 0 are required. Out-of-range values such as lr=0 are not intercepted. The implementation does not promise numerical consistency with the TensorFlow Golden, and only robustness tests (no crash, no out-of-bounds access) are performed.
- Intermediate computations such as power operations are performed in FLOAT32 and then downgraded to the input type.
- Attribute constraint: use_locking supports only false (the NPU does not have Ref semantics).
- Empty tensors are supported (the shape contains a 0 dimension, for example [0, N]), and the operator returns empty outputs through short-circuit.
- Dynamic shapes (-1 unknown dimensions) and dynamic ranks (-2 unknown rank) are supported.
- The operator is registered only on the Ascend 950PR&950DT products, and other products are not supported.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:---------|:---------|:-----|
| Graph-mode invocation | [test_geir_inplace_apply_ftrl_v2_dynamic](./examples/test_geir_inplace_apply_ftrl_v2_dynamic.cpp) | Calls the InplaceApplyFtrlV2 operator by building a graph through the [operator IR](./op_graph/inplace_apply_ftrl_v2_proto.h), covering the dynamic shape (-1/-2) scenarios. |
