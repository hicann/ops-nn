# ThnnFusedLstmCellGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                                            | Support |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT products</term>                                                |    √     |
| <term>Atlas A3 products</term>                         |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                                          |    ×     |
| <term>Atlas inference products</term>                                                 |    ×     |
| <term>Atlas training products</term>                                                  |    ×     |

## Function

- Operator function: Backward propagation of the remaining computation after matmul in the four gates of LSTMCell. It computes the gradients of the pre-activation gate values gates (forward output), the input cx, and the bias b.

- Formula:

**Variable definitions**

* **Input gradients**: $\delta h_t$ (`gradHy`), $\delta c_t$ (`gradC`)
* **Forward cache**: $i，f，g，o$ (activation values of each gate, `storage`), $c_{t-1}$ (`cx`), $c_t$ (`cy`)
* **Output gradients**: $\delta a_i，\delta a_f，\delta a_g，\delta a_o$ (stored in `gradInGatesOut`), $\delta c_{t-1}$ (`gradCPrevOut`)

**Phase 1: Intermediate gradients and state back-propagation**

First, compute the contribution of the hidden state to the cell state, and aggregate it to obtain the total gradient of the cell at the current time step $\text{grad\_}c_{total}$:

$$
\begin{aligned}
gcx &= \tanh(c_t) \\
\text{grad\_}c_{total} &= \delta h_t \cdot o \cdot (1 - gcx^2) + \delta c_t \\
\delta c_{t-1} &= \text{grad\_}c_{total} \cdot f
\end{aligned}
$$

**Phase 2: Gating component gradients (Pre-activation)**

According to the code logic, the gradients $\delta a$ of each gate before entering the activation function are computed as follows:

$$
\begin{aligned}
\delta a_o &= (\delta h_t \cdot gcx) \cdot o \cdot (1 - o) \\
\delta a_i &= (\text{grad\_}c_{total} \cdot g) \cdot i \cdot (1 - i) \\
\delta a_f &= (\text{grad\_}c_{total} \cdot c_{t-1}) \cdot f \cdot (1 - f) \\
\delta a_g &= (\text{grad\_}c_{total} \cdot i) \cdot (1 - g^2)
\end{aligned}
$$

**Phase 3: Parameter gradient (db)**

**1. Bias gradient (db):** Perform summation over the batch dimension ($N$):

$$
\delta b = \sum_{n=1}^{N} \begin{bmatrix} \delta a_i \\ \delta a_f \\ \delta a_g \\ \delta a_o \end{bmatrix}_n
$$

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>dhy</td>
      <td>Input</td>
      <td><ul><li>Gradient of the hidden state output by LSTMCell in the forward pass.</li><li>The shape is [batch, hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dc</td>
      <td>Input</td>
      <td><ul><li>Gradient of the cell state output by LSTMCell in the forward pass.</li><li>The shape is [batch, hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>Input</td>
      <td><ul><li>Cell state input to LSTMCell in the forward pass.</li><li>The shape is [batch, hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Input</td>
      <td><ul><li>Cell state output by LSTMCell in the forward pass.</li><li>The shape is [batch, hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>Input</td>
      <td><ul><li>Activation values of the four gates output by LSTMCell in the forward pass.</li><li>The shape is [batch, 4 * hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgates</td>
      <td>Output</td>
      <td><ul><li>Gradients of the pre-activation values of the four gates in the forward pass of LSTMCell.</li><li>The shape is [batch, 4 * hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dc_prev</td>
      <td>Output</td>
      <td><ul><li>Gradient of the cell state input in the forward pass of LSTMCell.</li><li>The shape is [batch, hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db</td>
      <td>Output</td>
      <td><ul><li>Gradient of the input bias in the forward pass of LSTMCell.</li><li>The shape is [4 * hidden_size].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>has_bias</td>
      <td>Attribute</td>
      <td>Whether the bias gradient needs to be computed.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_thnn_fused_lstm_cell_grad.cpp](examples/test_aclnn_thnn_fused_lstm_cell_grad.cpp) | Calls the ThnnFusedLstmCellGrad operator through the [aclnnThnnFusedLstmCellBackward](docs/aclnnThnnFusedLstmCellBackward_en.md) API. |      |
