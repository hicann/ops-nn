# DynamicRnn

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|×|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|×|

## Function

- Operator function: A basic recurrent neural network (Recurrent Neural Network) operator used to process sequence data. It transfers temporal information through the hidden state and is suitable for processing data with temporal/sequential dependencies.
- Operator formula: Taking the LSTM format as an example, given the input $x_t$, the hidden state $h_{t-1}$ at the previous time step, and the cell state $c_{t-1}$, the computation process of DynamicRNN is as follows:

    1. **Forget gate** (Forget Gate):
       $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

    2. **Input gate** (Input Gate):
       $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
       $$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

    3. **Cell state update**:
       $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

    4. **Output gate** (Output Gate):
       $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
       $$h_t = o_t \odot \tanh(c_t)$$

    Here:
    - $\sigma$ is the sigmoid function
    - $\odot$ denotes element-wise multiplication (Hadamard product)
    - $W_*$ is the learnable weight matrix
    - $b_*$ is the learnable bias term

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 470px">
  <col style="width: 170px">
  <col style="width: 50px">
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
      <td>Input data (T, Batch, input_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w</td>
      <td>Input</td>
      <td>Input weight (input_size + hidden_size, 4 * hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Input bias (4 * hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_length</td>
      <td>Input</td>
      <td>The actual length of each batch; the dimensions are (T, Batch, input_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>Input</td>
      <td>The hidden state at the initial time step; the dimensions are (1, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_c</td>
      <td>Input</td>
      <td>The cell state at the initial time step; the dimensions are (1, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>Input</td>
      <td>The mask matrix of the input.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_c</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>i</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>j</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>o</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>tanhc</td>
      <td>Output</td>
      <td>Output result; the dimensions are (T, batch_size, hidden_size).</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>cell_type</td>
      <td>Attribute</td>
      <td>The default is "LSTM", and currently only "LSTM" is implemented.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>Attribute</td>
      <td>The default is "UNIDIRECTIONAL".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_depth</td>
      <td>Attribute</td>
      <td>The number of levels of multi_rnn; the default is 1, and currently only 1 is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_peephole</td>
      <td>Attribute</td>
      <td>Specifies whether to use peephole; the default is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>Attribute</td>
      <td>The default is 1.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>Attribute</td>
      <td>The default value is -1.0 (no clipping).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_proj</td>
      <td>Attribute</td>
      <td>The dimension after dimensionality reduction by projection; the default 0 means no dimensionality reduction.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>time_major</td>
      <td>Attribute</td>
      <td>The arrangement of the input data; the default is True.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation</td>
      <td>Attribute</td>
      <td>The activation function "tanh".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>forget_bias</td>
      <td>Attribute</td>
      <td>The default is 0.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>Attribute</td>
      <td>The default is "ijfo".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Attribute</td>
      <td>The default is true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Atlas inference products: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | [test_geir_dynamic_rnn](examples/test_geir_dynamic_rnn.cpp)  | Calls the DynamicRnn operator by building a graph through the [operator IR](op_graph/dynamic_rnn_proto.h).         |
