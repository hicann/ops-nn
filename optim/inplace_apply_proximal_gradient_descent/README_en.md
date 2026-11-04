# InplaceApplyProximalGradientDescent

English | [简体中文](./README.md)

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

InplaceApplyProximalGradientDescent performs one element-wise proximal gradient update with L1 and L2 regularization on dense variables. This operator provides only the GE graph mode interface and returns the update result through the independent output `var_out`.

For each element at the same position in `var` and `delta`, first compute:

$$
prox\_v_i = var_i - alpha \times delta_i
$$

Then select the branch according to the shared scalar `l1`:

$$
var\_out_i =
\begin{cases}
\dfrac{\operatorname{sign}(prox\_v_i) \times
\max\left(\left|prox\_v_i\right| - alpha \times l1, 0\right)}
{1 + alpha \times l2}, & l1 > 0, \\
\dfrac{prox\_v_i}{1 + alpha \times l2}, & l1 \le 0.
\end{cases}
$$

Here $\operatorname{sign}(0)=0$, and `alpha`, `l1`, and `l2` are shared by all elements.

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
<td>Variable tensor to be updated. It is the deduction source of the output shape and data type.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>alpha</td>
<td>Input</td>
<td>Learning rate shared by all elements. Only the 0-D shape or [1] is supported.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l1</td>
<td>Input</td>
<td>L1 parameter shared by all elements. Only the 0-D shape or [1] is supported. Its value determines the computation branch.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>l2</td>
<td>Input</td>
<td>L2 parameter shared by all elements. Only the 0-D shape or [1] is supported.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>delta</td>
<td>Input</td>
<td>Dense increment involved in the element-wise update. Its shape must be the same as var dimension by dimension.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>var_out</td>
<td>Output</td>
<td>Updated variable tensor. Its shape and data type are both the same as var.</td>
<td>BF16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>use_locking</td>
<td>Optional attribute</td>
<td>Resource locking compatibility attribute. It does not change the computation formulas. The default value is false.</td>
<td>BOOL</td>
<td>-</td>
</tr>
</tbody>
</table>

## Constraints

- The rank range of `var` and `delta` is 0 to 16. The two must have exactly the same shape dimension by dimension, and broadcast is not supported.
- Each of `alpha`, `l1`, and `l2` supports only 0-D or `[1]`.
- The five inputs must use the same data type. Only three combinations are supported: all BF16, all FLOAT16, or all FLOAT.
- All inputs and outputs support only the ND data format.
- `var_out` is an independent output, and reusing the physical address with `var` is not guaranteed.
- `alpha`, `l1`, and `l2` have no positive or negative value restrictions. When `1 + alpha * l2` is zero or negative, the floating-point division is still performed.

## Invocation Description

|Invocation method|Sample code|Description|
|:---|:---|:---|
|GE graph mode|[test_geir_inplace_apply_proximal_gradient_descent.cpp](examples/arch35/test_geir_inplace_apply_proximal_gradient_descent.cpp)|Builds a graph through the [operator IR](op_graph/inplace_apply_proximal_gradient_descent_proto.h) and verifies the output shape, data type, and values.|
