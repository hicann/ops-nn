# ThnnFusedLstmCell

English | [简体中文](./README.md)

## Applicable Products

| Product                                                                            | Support |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT products</term>                                                |     √    |
| <term>Atlas A3 products</term>                         |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                                          |    ×     |
| <term>Atlas inference products</term>                                                 |    ×     |
| <term>Atlas training products</term>                                                  |    ×     |

## Function

- Description: Implements the subsequent computation after matrix multiplication in the single-step forward computation of a Long Short-Term Memory unit (LSTM Cell).

- Formula:

  Computes the gate activation values:

  $$
  \begin{aligned}
  b &= b_{ih} + b_{hh} \\
  gates &= inputGates + hiddenGates + b \\
  i_{out} &= \sigma(gates_{i}) \\
  g_{out} &= \tanh(gates_{g}) \\
  f_{out} &= \sigma(gates_{f}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$

  Updates the cell state:

  $$
  cy = f_{out} \odot cx + i_{out} \odot g_{out}
  $$

  Updates the hidden state:

  $$
  \begin{aligned}
  tanhc &= \tanh(cy) \\
  hy &= o_{out} \odot tanhc
  \end{aligned}
  $$

  Description of related symbols:

  * Biases $b_{ih} = \text{inputBias}$, $b_{hh} = \text{hiddenBias}$
  * $gates$ is evenly split along the last dimension into 4 components, that is, $gates \xrightarrow{\text{split}} [gates_i, gates_g, gates_f, gates_o]$
  * The obtained 4 gate activation values are concatenated along the last dimension into $\text{storage}$, that is, $[i_{out}, g_{out}, f_{out}, o_{out}] \xrightarrow{\text{concat}} \text{storage}$
  * $\sigma$ is the Sigmoid activation function, and $\odot$ is the element-wise product

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 130px">
  <col style="width: 144px">
  <col style="width: 400px">
  <col style="width: 200px">
  <col style="width: 100px">
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
      <td>inputGates</td>
      <td>Input</td>
      <td>The 4 gates of the input layer, that is, the values of the input gate (Input Gate), candidate cell state (Cell Candidate), forget gate (Forget Gate), and output gate (Output Gate).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenGates</td>
      <td>Input</td>
      <td>The values of the 4 gates of the hidden layer.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>Input</td>
      <td>The cell state at the previous time step.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>inputBias</td>
      <td>Input</td>
      <td>Input bias.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hiddenBias</td>
      <td>Input</td>
      <td>Hidden layer bias.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>Output</td>
      <td>The hidden state at the current time step, that is, the output at the current time step.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Output</td>
      <td>The cell state at the current time step.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>Output</td>
      <td>The activation values of the 4 gates, provided for backward computation.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_thnn_fused_lstm_cell.cpp](examples/test_aclnn_thnn_fused_lstm_cell.cpp) | Invokes the ThnnFusedLstmCell operator through the [aclnnThnnFusedLstmCell](docs/aclnnThnnFusedLstmCell_en.md) interface. |
