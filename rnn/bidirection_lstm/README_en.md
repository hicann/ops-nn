# BidirectionLSTM

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: the LSTM (Long Short-Term Memory) network is a special recurrent neural network (RNN) model. This operator performs LSTM network computation, receives an input sequence and an initial state, and returns an output sequence and a final state.
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

  - $x_t ∈ R^{d}$: input vector of the LSTM cell.
  - $f_t ∈ (0, 1)^{h}$: activation vector of the forget gate.
  - $i_t ∈ (0, 1)^{h}$: activation vector of the input gate and update gate.
  - $o_t ∈ (0, 1)^{h}$: activation vector of the output gate.
  - $h_i ∈ (-1, 1)^{h}$: hidden state vector, also known as the output vector of the LSTM cell.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: cell input activation vector.
  - $c_t ∈ R^{h}$: cell state vector.
  - $W ∈ R^{h×d}, (U ∈ R^{h×h})∩(b ∈ R^{h})$: weight matrix and bias vector parameters to be learned during training.

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
      <td>Input vector of the LSTM cell, which is x in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initH</td>
      <td>Input</td>
      <td>Initialized hidden state, which is h in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>initC</td>
      <td>Input</td>
      <td>Initialized cell state, which is c in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wIh</td>
      <td>Input</td>
      <td>input-hidden weight, which is W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHh</td>
      <td>Input</td>
      <td>hidden-hidden weight, which is W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>bIhOptional</td>
      <td>Input</td>
      <td>input-hidden bias, which is b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhOptional</td>
      <td>Input</td>
      <td>hidden-hidden bias, which is b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>wIhReverseOptional</td>
      <td>Input</td>
      <td>Reverse input-hidden weight, which is W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>wHhReverseOptional</td>
      <td>Input</td>
      <td>Reverse hidden-hidden weight, which is W in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bIhReverseOptional</td>
      <td>Input</td>
      <td>Reverse input-hidden bias, which is b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>bHhReverseOptional</td>
      <td>Input</td>
      <td>Reverse hidden-hidden bias, which is b in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>Attribute</td>
      <td>Represents the number of LSTM layers. Currently only 1 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias</td>
      <td>Attribute</td>
      <td>Represents whether a bias exists.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst</td>
      <td>Attribute</td>
      <td>Represents whether the batch is the first dimension. Currently only false is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection</td>
      <td>Attribute</td>
      <td>Represents whether the mode is bidirectional.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Output vector of the LSTM cell.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputHOut</td>
      <td>Output</td>
      <td>Final hidden state, which is h in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
       <tr>
      <td>outputCOut</td>
      <td>Output</td>
      <td>Final cell state, which is c in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_bidirection_lstm](examples/arch20/test_aclnn_bidirection_lstm.cpp) | Invokes the BidirectionLSTM operator through the [aclnnBidirectionLSTM](docs/aclnnBidirectionLSTM_en.md) interface.    |
