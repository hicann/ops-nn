# ThnnFusedGruCell

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: it performs gated fused computation for a single time step of a GRU (Gated Recurrent Unit). The inputs are two gate pre-activations (the input side and the hidden side), the hidden state of the previous step, and two optional biases. One computation produces the new hidden state hy and the intermediate quantity storage reused by backward computation. It suits the frequent single-step computation in step-by-step inference or training of recurrent neural networks.
- Formula: the gate pre-activations are evenly split into three parts along the last dimension in the gate order [r, z, n], giving gi_r, gi_z, gi_n and gh_r, gh_z, gh_n. The two biases are split into b1_r, b1_z, b1_n and b2_r, b2_z, b2_n in the same way and broadcast along the batch dimension. hx is the hidden state of the previous step, B is the batch dimension size, and H is the hidden dimension size.

  $$
  rg = \frac{1}{1 + e^{-(gi_r + gh_r + b1_r + b2_r)}}
  $$

  $$
  zg = \frac{1}{1 + e^{-(gi_z + gh_z + b1_z + b2_z)}}
  $$

  $$
  ng = \tanh(gi_n + b1_n + rg \times (gh_n + b2_n))
  $$

  $$
  hy = ng + zg \times (hx - ng)
  $$

  $$
  storage = [rg \mid zg \mid ng \mid hx \mid (gh_n + b2_n)]
  $$

- Formula variables: hy is the new hidden state. storage is the intermediate quantity reused by backward; its five segments [rg, zg, ng, hx, gh_n+b2_n] are concatenated along the last dimension, and each segment has H columns. When input_bias and hidden_bias are absent, they are equivalent to all-zero biases.
- Precision description: FLOAT16 and BFLOAT16 inputs complete the sigmoid, tanh, and multiply-add computations at float32 intermediate precision, and the computation results are rounded back to the original data type. The output data type matches the input.

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
<td>input_gates</td>
<td>Input</td>
<td>The gate pre-activation on the input side, corresponding to gi_r, gi_z, gi_n in the formula, with shape (B, 3H).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hidden_gates</td>
<td>Input</td>
<td>The gate pre-activation on the hidden side, corresponding to gh_r, gh_z, gh_n in the formula, with shape (B, 3H).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hx</td>
<td>Input</td>
<td>The hidden state of the previous step, corresponding to hx in the formula, with shape (B, H).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>input_bias</td>
<td>Optional input</td>
<td>The bias on the input side, corresponding to b1_r, b1_z, b1_n in the formula, with shape (3H,). When absent, it is equivalent to an all-zero bias.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hidden_bias</td>
<td>Optional input</td>
<td>The bias on the hidden side, corresponding to b2_r, b2_z, b2_n in the formula, with shape (3H,). When absent, it is equivalent to an all-zero bias.</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>hy</td>
<td>Output</td>
<td>The new hidden state, corresponding to hy in the formula, with shape (B, H).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>storage</td>
<td>Output</td>
<td>The intermediate quantity reused by backward computation, corresponding to storage in the formula, with shape (B, 5H).</td>
<td>BFLOAT16, FLOAT16, FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- The data types of inputs and outputs must be the same, and only BFLOAT16, FLOAT16, and FLOAT are supported. Cross data type combinations are not supported, and other data types such as DOUBLE and INT64 are not supported.
- The shapes of input_gates and hidden_gates must be the same and be (B, 3H), and the shape of hx must be (B, H). They must satisfy input_gates.shape[1] == 3 × hx.shape[1]. When input_bias and hidden_bias are not empty, their element counts must be 3H and the two must be the same.
- The data formats of inputs and outputs support only ND.
- Dynamic shapes and dynamic ranks are supported: in graph mode, -1 in the shape denotes an unknown dimension and [-2] denotes an unknown rank. The actual shape and rank are derived at execution time and must still satisfy the shape and format constraints above.
- B=0 or H=0 (an empty tensor with numel 0) is a valid input, and the operator directly returns empty outputs.
- In the aclnn API, an absent optional bias is expressed as a null pointer and is equivalent to an all-zero bias. Inputs and outputs both support non-contiguous tensors (the interface layer automatically makes inputs contiguous, and the interface layer writes outputs back element by element in the declared layout).

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | - | Calls through the [aclnnThnnFusedGruCell](docs/aclnnThnnFusedGruCell.md) API, including the two-phase invocation of the aclnnThnnFusedGruCellGetWorkspaceSize and aclnnThnnFusedGruCell APIs. |
| GE graph mode | - | Builds the graph and calls it through the [operator IR definition](op_graph/thnn_fused_gru_cell_proto.h). |
