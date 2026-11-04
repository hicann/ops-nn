# SoftplusV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Softplus activation function. When $\beta \cdot x$ does not exceed the threshold, it outputs $\frac{1}{\beta}\log(1+\exp(\beta \cdot x))$; otherwise it directly outputs $x$.
- Formula:

$$
out = \begin{cases}
\displaystyle \frac{1}{\beta}\log(1+\exp(\beta \cdot x)), & \beta \cdot x \le threshold \\
x, & \beta \cdot x > threshold
\end{cases}
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
<td>x</td>
<td>Input</td>
<td>Input tensor of the Softplus computation, corresponding to x in the formula.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>Output</td>
<td>Output tensor of the Softplus computation, corresponding to out in the formula. Its shape is the same as x.</td>
<td>FLOAT, FLOAT16, BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>beta</td>
<td>Optional attribute</td>
<td>Beta in the formula, with the default value 1.0.</td>
<td>FLOAT</td>
<td>-</td>
</tr>
<tr>
<td>threshold</td>
<td>Optional attribute</td>
<td>Threshold in the formula, with the default value 20.0.</td>
<td>FLOAT</td>
<td>-</td>
</tr>
</tbody>
</table>

- <term>Atlas inference products</term> and <term>Atlas training products</term>: the BFLOAT16 data type is not supported.

## Constraints

- The maximum number of dimensions of input x is 8 (rank ≤ 8, consistent with the aclnn interface). When the number of dimensions exceeds 8, both InferShape and Tiling/GetWorkspaceSize refuse to execute.
- The length of a single dimension of input x does not exceed INT32_MAX. The total number of elements (the product of all dimensions) must be within the range representable by int64; on overflow, Tiling refuses to execute.
- Scalar (0-dimensional) input is processed as 1 element, and an empty tensor (with 0 elements) is returned directly.

## Invocation Description

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_softplus](./examples/test_aclnn_softplus.cpp) | Invokes the SoftplusV2 operator through the aclnnSoftplusGetWorkspaceSize and aclnnSoftplus interfaces. For the interface description, see [aclnnSoftplus](./docs/aclnnSoftplus_en.md). |
| GE graph mode | - | Invokes the SoftplusV2 operator by constructing the graph through REG_OP(SoftplusV2) in the [operator IR definition](./op_graph/softplus_v2_proto.h). |
