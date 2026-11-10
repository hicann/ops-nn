# InplaceApplyFtrl

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

InplaceApplyFtrl performs the single-step parameter update of the FTRL-Proximal (Follow The Regularized Leader) optimizer for online learning and large-scale recommendation model training in sparse feature scenarios. The operator reads the current weight `var`, the gradient square accumulator `accum`, the linear accumulator `linear`, and the gradient `grad` of the current step, and updates the three state quantities `var`, `accum`, and `linear` in place according to the FTRL formula. `lr`, `l1`, `l2`, and `lr_power` are scalar hyperparameters. Internal computation always uses fp32, and inputs and outputs can be bfloat16, float16, or float32.

The standard V1 update formula is:

$$
\begin{aligned}
accum_{new} &= accum + grad \cdot grad \\
\sigma &= accum_{new}^{-lr\_power} - accum^{-lr\_power} \\
linear_{new} &= linear + grad - \frac{\sigma}{lr} \cdot var \\
y &= \frac{accum_{new}^{-lr\_power}}{lr} + 2 \cdot l2 \\
x &= l1 \cdot sign(linear_{new}) - linear_{new} \\
var &= \begin{cases} x / y, & |linear_{new}| > l1 \\ 0, & |linear_{new}| \le l1 \end{cases} \\
accum &= accum_{new},\quad linear = linear_{new}
\end{aligned}
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
<td>Model weight variable w, a state quantity updated in place. Its shape is exactly the same as those of accum, linear, and grad.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>Input</td>
<td>Gradient square accumulator n (n ≥ 0), a state quantity updated in place.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>Input</td>
<td>Linear accumulator z, a state quantity updated in place.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>grad</td>
<td>Input</td>
<td>Gradient g of the current step, with the same shape and dtype as var.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>lr</td>
<td>Input</td>
<td>Learning rate η (η > 0), a rank-0 scalar broadcast to the shape of var at runtime.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>Input</td>
<td>L1 regularization strength λ₁ (λ₁ ≥ 0), a rank-0 scalar.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>Input</td>
<td>L2 regularization strength λ₂ (λ₂ ≥ 0), a rank-0 scalar.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>lr_power</td>
<td>Input</td>
<td>Learning rate decay power p (p ≤ 0, usually -0.5), a rank-0 scalar.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>var</td>
<td>Output</td>
<td>Updated weight w', with the same shape and dtype as the input var, written back to var in place.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>accum</td>
<td>Output</td>
<td>Updated gradient square accumulator n', with the same shape and dtype as the input accum, written back to accum in place.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>linear</td>
<td>Output</td>
<td>Updated linear accumulator z', with the same shape and dtype as the input linear, written back to linear in place.</td>
<td>bfloat16, float16, float32</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>Specifies whether to lock the update. Default: false. Currently only false is supported.</td>
<td>Bool</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The four tensors `var`, `accum`, `linear`, and `grad` must have exactly the same shape. Broadcasting between tensors is not supported.
- `lr`, `l1`, `l2`, and `lr_power` must be rank-0 scalars (0-d tensors; GE normalizes a 0-d tensor to the storage shape {1}, so a single-element tensor with the explicit shape {1} is also accepted), and their data types must be the same as that of `var`.
- The eight input ports must use the same data type. Mixed dtypes are not supported, and only the three data types bfloat16, float16, and float32 are supported.
- The tensor rank is in [0, 8].
- Only the ND data format is supported (the tiling stage validates the format of all inputs and outputs, and a non-ND format is rejected).
- `lr > 0`, `l1 ≥ 0`, `l2 ≥ 0`, and `lr_power ≤ 0`. The scalar hyperparameters are Tensor inputs whose data resides on the device side, and the host side cannot read their values. The caller must guarantee these value range constraints; when they are violated, the computation result is undefined.
- Internal computation always uses fp32. bfloat16 and float16 inputs are first promoted to fp32 for computation and then written back in the original dtype.
- `use_locking` currently supports only the default value false, and passing true causes an error in the tiling stage.
- The four inputs `var`, `accum`, `linear`, and `grad` and the three outputs `var`, `accum`, and `linear` are declared with AutoContiguous at the operator definition layer. Non-contiguous tensors (for example, obtained through slicing or transposition) are automatically converted to a contiguous layout by the framework before computation, so the caller does not need to make them contiguous manually.
- Empty tensors (numel=0) are supported: the tiling stage sets BlockDim=1 and returns directly, and the kernel skips the computation.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_inplace_apply_ftrl_dynamic.cpp](./examples/test_geir_inplace_apply_ftrl_dynamic.cpp) | Invokes the operator in GE graph mode. For the operator IR definition, see [operator IR](op_graph/inplace_apply_ftrl_proto.h). |
