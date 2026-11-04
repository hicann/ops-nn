# BidirectionLSTMV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | × |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | × |

## Function

- Operator function: the LSTM (Long Short-Term Memory) network is a special recurrent neural network (RNN) model. The operator performs LSTM network computation, receives an input sequence and the initial state, and returns an output sequence and the final state. Compared with [aclnnBidirectionLSTM](../bidirection_lstm/docs//aclnnBidirectionLSTM_en.md), it adds the scenario where sequences have different lengths.
- Formula:

  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$: the input vector of the LSTM cell.
  - $f_t ∈ (0, 1)^{h}$: the forget gate activation vector.
  - $i_t ∈ (0, 1)^{h}$: the input gate and update gate activation vector.
  - $o_t ∈ (0, 1)^{h}$: the output gate activation vector.
  - $h_i ∈ (-1, 1)^{h}$: the hidden state vector, also known as the output vector of the LSTM cell.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: the cell input activation vector.
  - $c_t ∈ R^{h}$: the cell state vector.
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$: the weight matrix and bias vector parameters to be learned during training.

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
      <td>x</td>
      <td>Input</td>
      <td>The input vector of the LSTM cell, the x in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initH</td>
      <td>Input</td>
      <td>The initialized hidden state, the h in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initC</td>
      <td>Input</td>
      <td>The initialized cell state, the c in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wIh</td>
      <td>Input</td>
      <td>The input-hidden weight, the W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHh</td>
      <td>Input</td>
      <td>The hidden-hidden weight, the W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>bIhOptional</td>
      <td>Input</td>
      <td>The input-hidden bias, the b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhOptional</td>
      <td>Input</td>
      <td>The hidden-hidden bias, the b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>wIhReverseOptional</td>
      <td>Input</td>
      <td>The reverse input-hidden weight, the W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHhReverseOptional</td>
      <td>Input</td>
      <td>The reverse hidden-hidden weight, the W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bIhReverseOptional</td>
      <td>Input</td>
      <td>The reverse input-hidden bias, the b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhReverseOptional</td>
      <td>Input</td>
      <td>The reverse hidden-hidden bias, the b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>batchSizeOptional</td>
      <td>Input</td>
      <td>The batch_size that actually takes part in the computation in each iteration.</td>
      <td>INT</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>Attribute</td>
      <td>Denotes the number of LSTM layers. Currently only 1 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias</td>
      <td>Attribute</td>
      <td>Denotes whether a bias exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst</td>
      <td>Attribute</td>
      <td>Denotes whether the batch is the first dimension. Currently only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection</td>
      <td>Attribute</td>
      <td>Denotes whether it is bidirectional.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>packed</td>
      <td>Attribute</td>
      <td>Denotes whether the input x is compressed. It takes effect only when batchSizeOptional is not empty.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>The output vector of the LSTM cell.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputHOut</td>
      <td>Output</td>
      <td>The final hidden state, the h in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputCOut</td>
      <td>Output</td>
      <td>The final cell state, the c in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_bidirection_lstmv2](examples/arch20/test_aclnn_bidirection_lstmv2.cpp) | Calls the BidirectionLSTMV2 operator through the [aclnnBidirectionLSTMV2](docs/aclnnBidirectionLSTMV2_en.md) API. |
