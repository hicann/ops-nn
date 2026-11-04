# BlockLstm

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Implements the unified TF BlockLSTM (V1+V2) forward computation. It computes the cell state and hidden state of the LSTM over T time steps, and outputs the input gate, forget gate, candidate state, and output gate of each time step.
- Formula:

  Compute the gate activation values:

  $$
  \begin{aligned}
  gates &= x \cdot w_{ix} + h_{prev} \cdot w_{hh} + b \\
  i_{out} &= \sigma(gates_{i}) \\
  f_{out} &= \sigma(gates_{f} + forgetBias) \\
  g_{out} &= \tanh(gates_{g}) \\
  o_{out} &= \sigma(gates_{o})
  \end{aligned}
  $$

  Update the cell state:

  $$
  \begin{aligned}
  ci_{out} &= g_{out} \\
  cs_{out} &= f_{out} \odot cs_{prev} + i_{out} \odot ci_{out}
  \end{aligned}
  $$

  If cellClip > 0, clip the cell state:

  $$
  cs_{out} = \min(\max(cs_{out}, -cellClip), cellClip)
  $$

  Update the hidden state:

  $$
  \begin{aligned}
  co_{out} &= \tanh(cs_{out}) \\
  h_{out} &= o_{out} \odot co_{out}
  \end{aligned}
  $$

  If usePeephole is true, peephole connections are introduced into the gate computation:

  $$
  \begin{aligned}
  i_{out} &= \sigma(gates_{i} + wci \odot cs_{prev}) \\
  f_{out} &= \sigma(gates_{f} + forgetBias + wcf \odot cs_{prev}) \\
  o_{out} &= \sigma(gates_{o} + wco \odot cs_{out})
  \end{aligned}
  $$

  - gate_order="ifco" corresponds to TF BlockLSTMV2; gate_order="icfo" corresponds to TF BlockLSTM (V1)
  - $\sigma$ is the Sigmoid activation function, and $\odot$ is the element-wise product

## Parameters

<table style="undefined;table-layout: fixed; width: 885px"><colgroup>
  <col style="width: 194px">
  <col style="width: 138px">
  <col style="width: 300px">
  <col style="width: 133px">
  <col style="width: 120px">
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
      <td>seq_len_max</td>
      <td>Input</td>
      <td>Maximum value of the actual sequence length, used to mask time steps beyond this length.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input sequence, that is, the input data of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev</td>
      <td>Input</td>
      <td>Initial cell state.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev</td>
      <td>Input</td>
      <td>Initial hidden state.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>Input</td>
      <td>Fused weight matrix, which contains input weights and hidden weights.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci</td>
      <td>Input</td>
      <td>Input gate peephole weight. Pass an all-zero vector when usePeephole is false.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf</td>
      <td>Input</td>
      <td>Forget gate peephole weight. Pass an all-zero vector when usePeephole is false.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco</td>
      <td>Input</td>
      <td>Output gate peephole weight. Pass an all-zero vector when usePeephole is false.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Bias.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>forget_bias</td>
      <td>Attribute</td>
      <td>Bias added to the forget gate. Default 0.0 (no distinction between V1/V2, consistent with operator registration).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>Attribute</td>
      <td>Clipping threshold of the cell state. No clipping is performed when the value is 0. Default 0.0 (no distinction between V1/V2, consistent with operator registration).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>Attribute</td>
      <td>Whether to use peephole connections.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>Attribute</td>
      <td>Gate order. "ifco" corresponds to TF BlockLSTMV2, and "icfo" corresponds to TF BlockLSTM (V1).</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>i</td>
      <td>Output</td>
      <td>Input gate activation value of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs</td>
      <td>Output</td>
      <td>Cell state of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Output</td>
      <td>Forget gate activation value of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>Output</td>
      <td>Output gate activation value of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ci</td>
      <td>Output</td>
      <td>Candidate state of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>co</td>
      <td>Output</td>
      <td>tanh(cs) value of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Output</td>
      <td>Hidden state of each time step.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Only the Ascend 950PR&950DT products are supported.
- The data types of all Input and Output parameters must be the same. FLOAT and FLOAT16 are supported.
- Input dimensions constraints: x supports only 3 dimensions with the shape (T, B, I); cs_prev/h_prev support only 2 dimensions with the shape (B, H); w is 2D (I+H, 4H); wci/wcf/wco are 1D (H); b is 1D (4H). The operator rejects execution when dimensions or shapes do not match (consistent with TF BlockLSTM/BlockLSTMV2: x must be 3D and cs_prev must be 2D).
- Zero-dimension (degenerate) support: T/B/H/I = 0 are all valid (consistent with TF; empty dimensions return empty tensors). When I == 0, the output remains non-empty with shape [T,B,H] (only the x-side projection is empty, and gates are still computed from h_prev and b); when T/B/H == 0, all outputs are zero-element tensors, the GE graph prunes the node after InferShape and does not start the kernel. Zero dimensions across inputs must be consistent (for example, when H=0, wci/wcf/wco/b must all be empty and w must be [I,0]); inconsistent values are rejected by shape validation. Only negative dimensions are rejected.
- Input value range constraint: Input values containing ±Inf or NaN are not supported. Inf/NaN inputs violate the finite value range defined by IEEE 754, and LSTM recursion (Inf × 0 = NaN) produces unpredictable output, which is not guaranteed to be bit-consistent with any reference implementation.
- seq_len_max must satisfy 0 <= seq_len_max <= T, and the output of time steps beyond T is set to zero. Out-of-range values visible on the host side (Const binding) are directly rejected in the tiling phase; dynamic values on the device side cannot be validated in tiling, and the kernel clamps them to [0, T] before computation (different from the error semantics of TF; callers must ensure the value is valid).
- seq_len_max supports only a scalar or a single-element tensor (rank 0 or [1]), and multiple elements (such as the seq_len[B] array of V1 in the TF 1.x era) are explicitly rejected in the InferShape phase. This is consistent with the public API contract of TF 2.6: seq_len_max of both BlockLSTM(V1) and BlockLSTMV2 must be a single element, and passing multiple elements causes a CHECK crash in TF itself (tested on TF 2.6.5). The unified interface does not provide the per-sample masking semantics of the historical V1 version, and callers who need per-sample variable-length output must mask the output themselves.
- When use_peephole is false, pass all-zero vectors for wci/wcf/wco.

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GEIR invocation | [test_geir_block_lstm](examples/arch35/test_geir_block_lstm.cpp) | Calls the BlockLstm operator in GEIR graph mode (static shape, including CPU golden value verification). |
| GEIR dynamic | [test_geir_block_lstm_dynamic](examples/arch35/test_geir_block_lstm_dynamic.cpp) | Runs scenarios with unknown dimensions (-1) and unknown rank (-2) continuously in the same graph and Session, and verifies the shape/dtype/value of all outputs. |
