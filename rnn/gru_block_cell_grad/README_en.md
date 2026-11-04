# GRUBlockCellGrad

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

GRUBlockCellGrad is the single-time-step backward gradient operator of the GRU block cell. It takes the upstream gradient d_h and the forward-cached gate intermediate quantities r, u, and c as inputs, computes the gradient d_x of the input x, the gradient d_h_prev of the hidden state h_prev at the previous time step, and the gate pre-activation gradient concatenation matrices d_c_bar and d_r_bar_u_bar. The operator completes the fused computation of two GEMM operations and the element-wise gradient chain within a single kernel. It serves the training backpropagation of GRU networks under TensorFlow graph mode and is used in pairs with the forward operator GRUBlockCell.

Formula (∘ is element-wise multiplication, × is matrix multiplication, [a | b] is concatenation by columns, and w_c^T and w_ru^T are the transposes of the weight matrices):

$$
d\_c\_bar = d\_h \circ (1-u) \circ (1 - c \circ c)
$$

$$
d\_u\_bar = d\_h \circ (h\_prev - c) \circ u \circ (1-u)
$$

$$
[d\_x\_component\_2 \mid d\_h\_prevr] = d\_c\_bar \times w\_c^{T}
$$

$$
d\_r\_bar = (d\_h\_prevr \circ h\_prev \circ r) \circ (1-r)
$$

$$
d\_r\_bar\_u\_bar = [d\_r\_bar \mid d\_u\_bar]
$$

$$
[d\_x\_component\_1 \mid d\_h\_prev\_component\_1] = d\_r\_bar\_u\_bar \times w\_ru^{T}
$$

$$
d\_x = d\_x\_component\_1 + d\_x\_component\_2
$$

$$
d\_h\_prev = d\_h\_prev\_component\_1 + d\_h\_prevr \circ r + d\_h \circ u
$$

Here input_size is the length of dimension 1 of x, and cell_size is the length of dimension 1 of h_prev; d_x_component_1 and d_x_component_2 take the first input_size columns of the two GEMM output matrices, and d_h_prev_component_1 and d_h_prevr take the last cell_size columns of the corresponding output matrices. The gradients of the weights and biases are not computed inside this operator; the upper-layer framework continues the computation using d_c_bar and d_r_bar_u_bar.

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
<td>Input of the current time step. Its shape defines batch and input_size; its values do not enter the computation chain and only participate in parameter validation, aligning with the competitor interface.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>h_prev</td>
<td>Input</td>
<td>Hidden state at the previous time step, corresponding to h_prev in the formula. Its shape defines batch and cell_size.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>w_ru</td>
<td>Input</td>
<td>Weights of the reset gate and the update gate, corresponding to w_ru in the formula, with shape (input_size+cell_size, 2*cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>w_c</td>
<td>Input</td>
<td>Candidate state weight, corresponding to w_c in the formula, with shape (input_size+cell_size, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>b_ru</td>
<td>Input</td>
<td>Biases of the reset gate and the update gate, with shape (2*cell_size,); the values do not enter the computation chain and only participate in parameter validation, aligning with the competitor interface.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>b_c</td>
<td>Input</td>
<td>Candidate state bias, with shape (cell_size,); the values do not enter the computation chain and only participate in parameter validation, aligning with the competitor interface.</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>r</td>
<td>Input</td>
<td>Forward reset gate output, corresponding to r in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>u</td>
<td>Input</td>
<td>Forward update gate output, corresponding to u in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>c</td>
<td>Input</td>
<td>Forward candidate state output, corresponding to c in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_h</td>
<td>Input</td>
<td>Upstream gradient of the new hidden state, corresponding to d_h in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_x</td>
<td>Output</td>
<td>Gradient of the input x, corresponding to d_x in the formula, with shape (batch, input_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_h_prev</td>
<td>Output</td>
<td>Gradient of the hidden state at the previous time step, corresponding to d_h_prev in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_c_bar</td>
<td>Output</td>
<td>Gradient of the candidate state pre-activation, corresponding to d_c_bar in the formula, with shape (batch, cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
<tr>
<td>d_r_bar_u_bar</td>
<td>Output</td>
<td>Column-wise concatenation of the pre-activation gradients of the reset gate and the update gate, corresponding to d_r_bar_u_bar in the formula, with shape (batch, 2*cell_size).</td>
<td>FLOAT</td>
<td>ND</td>
</tr>
</tbody>
</table>

## Constraints

- Only the FLOAT data type and the ND data format are supported.
- The input shapes are uniquely determined by the (batch, input_size, cell_size) triple. batch and input_size must be greater than or equal to 0, and cell_size must be greater than 0. The shape binding relationships of the tensors are x==(batch, input_size), h_prev==r==u==c==d_h==(batch, cell_size), w_ru==(input_size+cell_size, 2*cell_size), w_c==(input_size+cell_size, cell_size), b_ru==(2*cell_size,), and b_c==(cell_size,); the outputs are d_x==(batch, input_size), d_h_prev==d_c_bar==(batch, cell_size), and d_r_bar_u_bar==(batch, 2*cell_size). Parameter combinations with mismatched rank or shape are rejected during validation.

## Invocation Description

GE graph-mode invocation is provided; the TensorFlow parsing plugin maps `tf.raw_ops.GRUBlockCellGrad` to the GE operator of the same name.

| Invocation method | Sample code | Description |
| :--- | :--- | :--- |
| GE graph mode | [test_geir_gru_block_cell_grad.cpp](examples/arch35/test_geir_gru_block_cell_grad.cpp) | For the operator IR definition, see [op_graph/gru_block_cell_grad_proto.h](op_graph/gru_block_cell_grad_proto.h). |
