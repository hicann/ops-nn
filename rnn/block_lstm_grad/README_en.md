# BlockLstmGrad

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

- Operator function: Implements the unified TF BlockLSTMGrad (V1+V2) backpropagation computation. Based on the forward inputs and the forward cache, it computes the gradients of the inputs, weights, bias, and gates of BlockLSTM.
- Formula:

  **Variable definitions**

  * **Forward inputs**: $x$ (`x` [T,B,I]), $cs_{prev}$ (`csPrev` [B,H]), $h_{prev}$ (`hPrev` [B,H]), $w$ (`w` [I+H,4H]), $b$ (`b` [4H])
  * **Forward cache**: $i, cs, f, o, ci, co, h$ (gate activation values and cell state [T,B,H])
  * **Gradient inputs**: $\delta cs$ (`csGrad` [T,B,H]), $\delta h$ (`hGrad` [T,B,H])
  * **Output gradients**: $\delta x$ (`xGradOut`), $\delta cs_{prev}$ (`csPrevGradOut`), $\delta h_{prev}$ (`hPrevGradOut`), $\delta w$ (`wGradOut`), $\delta wci$ (`wciGradOut`), $\delta wcf$ (`wcfGradOut`), $\delta wco$ (`wcoGradOut`), $\delta b$ (`bGradOut`)

  **Backward computation process**

  Let $co_t = \tanh(cs_t)$ (that is, the forward cache `co`). $W_x$ and $W_h$ are the first $I$ rows and the last $H$ rows of $w$, respectively. Recurse backward along time steps from $t=T-1$ to $t=0$, and pass back the cell state gradient and hidden state gradient step by step:

  $$
  \begin{aligned}
  \delta cs_t^{tot} &= \delta h_t \cdot o_t \cdot (1 - co_t^2) + \delta cs_t + \delta cs_{t+1}^{tot} \cdot f_{t+1} \\
  \delta a_i &= \delta cs_t^{tot} \cdot ci_t \cdot i_t \cdot (1 - i_t) \\
  \delta a_f &= \delta cs_t^{tot} \cdot cs_{t-1} \cdot f_t \cdot (1 - f_t) \\
  \delta a_g &= \delta cs_t^{tot} \cdot i_t \cdot (1 - ci_t^2) \\
  \delta a_o &= \delta h_t \cdot co_t \cdot o_t \cdot (1 - o_t)
  \end{aligned}
  $$

  Boundary: when $t=T-1$, the $\delta cs_{t+1}^{tot}$ term does not exist (treat it as 0); when $t=0$, $cs_{-1} = cs_{prev}$.

  Output and parameter gradients of each time step ($\delta a$ denotes the preceding four pre-activation gradients):

  $$
  \begin{aligned}
  \delta x_t &= \sum_{gate \in \{i,f,g,o\}} \delta a_{gate} \cdot W_x[gate] \\
  \delta h_{t-1} &= \sum_{gate \in \{i,f,g,o\}} \delta a_{gate} \cdot W_h[gate] \quad (t=0 时即 \delta h_{prev}) \\
  \delta cs_{prev} &= \delta cs_0^{tot} \cdot f_0 \\
  \delta b &= \sum_{t,b} \delta a, \quad \delta w = \sum_{t,b} \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} \cdot \delta a
  \end{aligned}
  $$

  For $t>0$, $\delta h_{t-1}$ is merged into the $\delta h$ of the next time step (that is, time step $t-1$) and continues to be passed back. When use_peephole is true, the pre-activation of each gate additionally contains peephole terms ($wci \odot cs_{t-1}$, $wcf \odot cs_{t-1}$, $wco \odot co_t$), and $\delta wci/\delta wcf/\delta wco$ are accumulated over $(t,b)$ in the same way.

  The implementation of $\delta w$ and $\delta b$ first writes workspace partials by cluster over the time-step and Batch dimensions, and then reduces and sums them in a fixed order (deterministic result).

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
      <td>Maximum sequence length.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Forward input of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev</td>
      <td>Input</td>
      <td>Initial value of the cell state input in the BlockLSTM forward direction.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev</td>
      <td>Input</td>
      <td>Initial value of the hidden state input in the BlockLSTM forward direction.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>Input</td>
      <td>Forward weight matrix of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci</td>
      <td>Input</td>
      <td>Peephole weight ci.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf</td>
      <td>Input</td>
      <td>Peephole weight cf.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco</td>
      <td>Input</td>
      <td>Peephole weight co.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Forward bias of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>i</td>
      <td>Input</td>
      <td>Forward input gate activation value of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs</td>
      <td>Input</td>
      <td>Forward cell state of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Input</td>
      <td>Forward forget gate activation value of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>Input</td>
      <td>Forward output gate activation value of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ci</td>
      <td>Input</td>
      <td>Forward candidate state of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>co</td>
      <td>Input</td>
      <td>Forward tanh(cs) value of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Input</td>
      <td>Forward hidden state of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_grad</td>
      <td>Input</td>
      <td>Gradient of the forward output cell state of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_grad</td>
      <td>Input</td>
      <td>Gradient of the forward output hidden state of BlockLSTM.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>Attribute</td>
      <td>Whether to use peephole.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>Attribute</td>
      <td>Gate order. "ifco" and "icfo" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x_grad</td>
      <td>Output</td>
      <td>Gradient of input x.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cs_prev_grad</td>
      <td>Output</td>
      <td>Gradient of the initial cell state input.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>h_prev_grad</td>
      <td>Output</td>
      <td>Gradient of the initial hidden state input.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_grad</td>
      <td>Output</td>
      <td>Gradient of the weight matrix w. It is written directly by deterministic reduction, and callers do not need to zero-initialize it.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wci_grad</td>
      <td>Output</td>
      <td>Gradient of peephole weight ci.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wcf_grad</td>
      <td>Output</td>
      <td>Gradient of peephole weight cf.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>wco_grad</td>
      <td>Output</td>
      <td>Gradient of peephole weight co.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b_grad</td>
      <td>Output</td>
      <td>Gradient of bias b. It is written directly by deterministic reduction, and callers do not need to zero-initialize it.</td>
      <td>FLOAT/FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The data types of all Input and Output parameters must be FLOAT or FLOAT16 and must be the same.
- Only the Ascend 950PR&950DT products are supported.
- w_grad and b_grad are written directly by deterministic reduction, and callers do not need to zero-initialize them.
- gate_order supports two gate orders: "ifco" (default) and "icfo".
- seq_len_max must satisfy 0 <= seq_len_max <= T and supports only a scalar or a single-element tensor (rank 0 or [1]); multiple elements (such as the seq_len[B] array of V1 in the TF 1.x era) are explicitly rejected in the InferShape phase — consistent with the public API contract of TF 2.6 (both BlockLSTM(V1)/BlockLSTMV2 require a single element, and multiple elements cause a CHECK crash in TF itself, as tested).

## Invocation Description

| Invocation method | Invocation sample | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| GEIR invocation | [test_geir_block_lstm_grad](examples/arch35/test_geir_block_lstm_grad.cpp) | Calls the BlockLstmGrad operator in GEIR graph mode (fixed shape). |
| GEIR dynamic shape invocation | [test_geir_block_lstm_grad_dynamic](examples/arch35/test_geir_block_lstm_grad_dynamic.cpp) | Scenarios with dynamic shape (-1) and dynamic rank (-2); the same graph runs with multiple sets of shapes and performs double-precision golden verification. |
