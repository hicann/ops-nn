# LSTMBlockCellGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

LSTMBlockCellGrad computes the backward gradients of the TensorFlow LSTMBlockCell unit (including peephole connections). From cs_grad and h_grad of the current time step, combined with the forward gating intermediate quantities i/cs/f/o/ci/co, it reconstructs the previous cell state gradient cs_prev_grad, the fused gate pre-activation gradient dicfo (column blocks [di, dc, df, do], with the icfo column order consistent with the forward fused weight w), and the three peephole weight gradients wci_grad/wcf_grad/wco_grad. The semantics are fully aligned with TensorFlow tf.raw_ops.LSTMBlockCellGrad (16 inputs/5 outputs/the use_peephole attribute are all identical item by item). The operator is called in GE graph mode, and tf_plugin claims the TF front-end graph construction nodes.

When use_peephole=false, the three peephole weight gradient outputs are exactly all 0 (consistent with the native TF behavior). x/h_prev/w/b do not enter the gradient formulas: their gradients are reconstructed outside the operator by the caller through dicfo (matrix multiplication), and this operator participates only in the shape contract validation.

Formula (symbol conventions: $\odot$ denotes element-wise multiplication; $\sum_{b=0}^{B-1}$ denotes summation along the batch dimension (axis 0); $i$ / $f$ / $o$ use the sigmoid activation, and $ci$ / $co$ use the tanh activation; $B$ = batch, $C$ = cell, $N$ = num_inputs):

$$
dig = 1 - co \odot co, \qquad do_{pre} = h_{grad} \odot co \odot o \odot (1 - o)
$$

$$
dcs = cs_{grad} + h_{grad} \odot o \odot dig + \begin{cases} do_{pre} \odot wco, & \text{use\_peephole} = \text{true} \\ 0, & \text{use\_peephole} = \text{false} \end{cases}
$$

$$
di_{pre} = dcs \odot ci \odot i \odot (1 - i), \quad dci_{pre} = dcs \odot i \odot (1 - ci \odot ci), \quad df_{pre} = dcs \odot cs_{prev} \odot f \odot (1 - f)
$$

$$
cs_{prev\_grad} = dcs \odot f + \begin{cases} wci \odot di_{pre} + wcf \odot df_{pre}, & \text{use\_peephole} = \text{true} \\ 0, & \text{use\_peephole} = \text{false} \end{cases}
$$

$$
dicfo = \big[\, di_{pre} \;\big|\; dci_{pre} \;\big|\; df_{pre} \;\big|\; do_{pre} \,\big]
$$

$$
wci_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs_{prev}[b,:] \odot di_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases} \quad wcf_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs_{prev}[b,:] \odot df_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases} \quad wco_{grad} = \begin{cases} \sum_{b=0}^{B-1} cs[b,:] \odot do_{pre}[b,:], & \text{true} \\ \mathbf{0}, & \text{false} \end{cases}
$$

## Parameters

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 420px">
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
<tr><td>x</td><td>Input</td><td>Cell input tensor, with the shape (batch, num_inputs).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>cs_prev</td><td>Input</td><td>Cell state of the previous time step, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>h_prev</td><td>Input</td><td>Output of the previous time step, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>w</td><td>Input</td><td>Weight matrix, with the shape (num_inputs+cell, 4*cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wci</td><td>Input</td><td>Input gate peephole weight, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wcf</td><td>Input</td><td>Forget gate peephole weight, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wco</td><td>Input</td><td>Output gate peephole weight, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>b</td><td>Input</td><td>Bias vector, with the shape (4*cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>i</td><td>Input</td><td>Forward input gate activation value, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>cs</td><td>Input</td><td>Cell state before the forward tanh, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>f</td><td>Input</td><td>Forward forget gate activation value, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>o</td><td>Input</td><td>Forward output gate activation value, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>ci</td><td>Input</td><td>Forward cell gate activation value, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>co</td><td>Input</td><td>Output after the forward tanh, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>cs_grad</td><td>Input</td><td>Gradient of cs at the current time step, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>h_grad</td><td>Input</td><td>Gradient of h at the current time step, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>use_peephole</td><td>Attribute</td><td>Whether to use diagonal peephole connections, of the bool type, with the default value false. When it is false, the three peephole weight gradient outputs are exactly all 0.</td><td>BOOL</td><td>-</td></tr>
<tr><td>cs_prev_grad</td><td>Output</td><td>Gradient of cs_prev, with the shape (batch, cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>dicfo</td><td>Output</td><td>Concatenation of the four gate gradients di, dc, df, and do, with the shape (batch, 4*cell).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wci_grad</td><td>Output</td><td>Input gate peephole weight gradient, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wcf_grad</td><td>Output</td><td>Forget gate peephole weight gradient, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
<tr><td>wco_grad</td><td>Output</td><td>Output gate peephole weight gradient, with the shape (cell,).</td><td>FLOAT, FLOAT16</td><td>ND</td></tr>
</tbody>
</table>

## Constraints

- The 16 tensor inputs must share the same data type (FLOAT or FLOAT16). Mixed dtype combinations are rejected during the GEIR execution phase.
- The batch dimension must be consistent (x is batch×num_inputs, and the other 2D inputs are batch×cell); w.shape must be (num_inputs+cell, 4*cell); b.shape must be 4*cell; wci/wcf/wco.shape must be cell.
- Only the ND format is supported.
- Empty tensors (batch=0 or cell=0) are valid inputs, and the outputs have the corresponding empty shapes.
- The caller must ensure that the value of use_peephole is consistent with the TF front-end graph construction (the operator side does not validate the value range of this attribute).
- Determinism statement: the implementation is deterministic by default, and multiple executions with the same shape and the same input produce bit-level identical results.

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
<td><a href="./examples/test_geir_lstm_block_cell_grad.cpp">test_geir_lstm_block_cell_grad</a></td>
<td>Calls the LSTMBlockCellGrad operator by constructing a graph through the operator IR.</td>
</tr>
</tbody>
</table>
