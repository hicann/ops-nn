# InplaceApplyRMSProp

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----------------------------------------|:------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

`InplaceApplyRMSProp` is the parameter update operator of the RMSProp adaptive learning rate optimizer. The
operator maintains a sliding root mean square and a momentum buffer of the gradient, and updates the three
state tensors `var`, `ms`, and
`mom` in each iteration. The three explicit outputs return the updated results of the inputs with the same names respectively.

This operator currently only provides the AscendC GE graph-mode path and does not provide an independent `aclnn` interface.

The calculation formula is as follows:

$$
\begin{aligned}
\text{ms}_t &= \rho \cdot \text{ms}_{t-1} + (1 - \rho) \cdot \text{grad}^2 \\
\text{mom}_t &= \text{momentum} \cdot \text{mom}_{t-1} +
\frac{\text{lr} \cdot \text{grad}}{\sqrt{\text{ms}_t + \epsilon}} \\
\text{var}_t &= \text{var}_{t-1} - \text{mom}_t
\end{aligned}
$$

In the formula, `lr` is the learning rate, `rho` is the decay coefficient, `momentum` is the momentum coefficient, and `epsilon`
is the numerical stability constant.

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
<td>Weight tensor to be updated, corresponding to var in the formula, updated in place. Its shape and dtype are the same as those of ms, mom, and grad.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>ms</td>
<td>Input</td>
<td>Sliding average state of the squared gradient, corresponding to ms in the formula, updated in place. You must ensure that it is non-negative, and the implementation performs no numerical validation.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mom</td>
<td>Input</td>
<td>Momentum buffer, corresponding to mom in the formula, updated in place.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>Learning rate, corresponding to lr in the formula. It is a 1D ND scalar tensor with shape {1}.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>rho</td>
<td>Input</td>
<td>Decay coefficient, corresponding to rho in the formula. It is a 1D ND scalar tensor with shape {1}. The recommended value is [0, 1), and the implementation performs no numerical validation.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>momentum</td>
<td>Input</td>
<td>Momentum coefficient, corresponding to momentum in the formula. It is a 1D ND scalar tensor with shape {1}.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>epsilon</td>
<td>Input</td>
<td>Numerical stability constant, corresponding to epsilon in the formula. It is a 1D ND scalar tensor with shape {1}.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>Current gradient tensor, corresponding to grad in the formula. Its shape and dtype are the same as those of var.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>Output</td>
<td>Updated parameter, corresponding to the input var, written back to var in place.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>ms</td>
<td>Output</td>
<td>Updated sliding average of the squared gradient, corresponding to the input ms, written back to ms in place.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mom</td>
<td>Output</td>
<td>Updated momentum buffer, corresponding to the input mom, written back to mom in place.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>Whether to lock during the update. The default value is false. The current implementation serves only as a compatibility placeholder and introduces no additional mutex.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The shapes and dtypes of `var`, `ms`, `mom`, and `grad` must be exactly the same, and rank1-8 is supported. You must ensure that `ms` is non-negative, and the implementation performs no numerical validation. A negative value may cause the square root computation to produce NaN.
- Input shape markers `-1` (unknown dimension) and `-2` (unknown rank) are registered as dynamic shape/rank and support shape inference propagation. However, the execution path requires the final shape to resolve to a concrete ND shape with rank∈[1,8] and non-negative dimensions. Unresolved `-1`/`-2` are rejected at the Tiling stage.
- Empty tensors apply only to the state tensors: when `var` (and `ms`, `mom`, and `grad` whose shapes match it) has `numel == 0`, the empty path is taken and no update is executed. `lr`, `rho`, `momentum`, and `epsilon` must still have shape {1}, and scalar inputs with `numel == 0` are rejected.
- `lr`, `rho`, `momentum`, and `epsilon` must be ND scalar tensors with shape{1}, and their dtype must be the same as that of `var`. The recommended value of `rho` is [0, 1), and the implementation performs no range validation.
- Non-contiguous tensors (view strides/offset) are not declared as supported: you must ensure that `var`, `ms`, `mom`, and `grad` and the four scalar inputs use contiguous memory (ND format). Otherwise, the behavior is undefined.
- Invalid inputs are rejected according to the following rules: a rank outside [1,8] reports `EZ0012`; inconsistent state tensor shapes report `EZ0010`; a scalar shape that is not {1}, a negative dimension, or an invalid element count reports `EZ0009`; an invalid `var` dtype reports `EZ0020`, and inconsistent dtypes across inputs report `EZ0021`.
- `epsilon` is processed as FP32 in the computation domain. When its value is less than or equal to `FLT_MIN` or is NaN, the kernel uses `FLT_MIN` as the lower bound.
- For the `FLOAT` and `BFLOAT16` paths, `grad²` is computed in the FP32 domain. The two have the same exponent range. When `|grad| > sqrt(FLT_MAX)≈1.84e19`, this intermediate quantity overflows to `inf` in FP32, and `ms`, `mom`, and `var` may deviate from the mathematically defined values. For `FLOAT16`, `|grad| ≤ 65504`, so this overflow does not exist.
- `FLOAT`, `FLOAT16`, and `BFLOAT16` are supported. The FP16/BF16 paths perform the computation in the FP32 domain and then convert the result back to the target dtype.
- `use_locking` does not change the calculation result, and the current implementation introduces no additional mutex.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_inplace_apply_rms_prop](./examples/test_geir_inplace_apply_rms_prop.cpp) | Invokes the operator in GE graph mode. For the operator IR definition, see [operator IR](./op_graph/inplace_apply_rms_prop_proto.h). |

Currently, no ACLNN, PyTorch, ONNX, or fusion rule interface is provided. The TensorFlow native RMSProp interface is inconsistent with the three-explicit-output prototype of this operator, so it does not serve as a valid invocation path.
