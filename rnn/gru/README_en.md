# Gru

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT products</term>                            |     √     |
| <term>Atlas A3 products</term>     |     √     |
| <term>Atlas A2 products</term> |     √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |     ×    |
| <term>Atlas inference products</term>                             |     ×     |
| <term>Atlas training products</term>                              |     ×   |

## Function

- Description: Implements the Gated Recurrent Unit (GRU) computation. It supports multi-layer stacking, bidirectional mode, and two input modes: fixed-length sequences and variable-length sequences (PackedSequence). In training mode, it can output the intermediate results of the gates (r, z, n, n_h) for backward propagation.

- Input and output shapes:
  - Fixed-length mode: the shape of the input x is (T, B, I) or (B, T, I), the shape of the output y is (T, B, D*H) or (B, T, D*H), and the shape of the final hidden state output_h is (L*D, B, H).
  - Variable-length mode: the shape of the input x is (sum(batch_size), I), the shape of the output y is (sum(batch_size), D*H), and the shape of the final hidden state output_h is (L*D, B, H). Currently, this mode is not supported.

- Formula:

  $$
  r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
  $$

  $$
  z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
  $$

  $$
  n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)} + b_{hn}))
  $$

  $$
  h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
  $$

  where:
  - $\sigma$ is the sigmoid activation function.
  - $\odot$ denotes element-wise multiplication.
  - $r_t$, $z_t$, and $n_t$ are the reset gate, update gate, and new gate respectively.
  - $h_t$ is the hidden state at the current time step.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :--- | :--- | :--- | :--- | :--- |
| x | Input | Denotes the input sequence data, corresponding to $x_t$ in the formula. In fixed-length mode, the shape is (T, B, I) or (B, T, I); in variable-length mode, the shape is (sum(batch_size), I). | FLOAT32, FLOAT16 | ND |
| wi | Input | Denotes the input weight matrix, corresponding to $W_{ir}$, $W_{iz}$, and $W_{in}$ in the formula. The shape is $[3H, I]$ (first layer) or $[3H, D*H]$ (non-first layer). | FLOAT32, FLOAT16 | ND |
| wh | Input | Denotes the hidden state weight matrix, corresponding to $W_{hr}$, $W_{hz}$, and $W_{hn}$ in the formula. The shape is $[3H, H]$. | FLOAT32, FLOAT16 | ND |
| bi | Optional input | Denotes the input bias, corresponding to $b_{ir}$, $b_{iz}$, and $b_{in}$ in the formula. The shape is $[3H]$. | FLOAT32, FLOAT16 | ND |
| bh | Optional input | Denotes the hidden state bias, corresponding to $b_{hr}$, $b_{hz}$, and $b_{hn}$ in the formula. The shape is $[3H]$. | FLOAT32, FLOAT16 | ND |
| batch_sizes | Optional input | Denotes the array of batch sizes of variable-length sequences, corresponding to the PackedSequence mode. The shape is (T). | INT64 | ND |
| init_h | Optional input | Denotes the initial hidden state, corresponding to $h_0$ in the formula. The shape is (L*D, B, H). | FLOAT32, FLOAT16 | ND |
| y | Output | Denotes the outputs of all time steps, corresponding to $h_t$ in the formula. In fixed-length mode, the shape is (T, B, D*H) or (B, T, D*H); in variable-length mode, the shape is (sum(batch_size), D*H). | FLOAT32, FLOAT16 | ND |
| output_h | Output | Denotes the final hidden states of all directions in the last layer, corresponding to the $h_t$ at the last time step in the formula. The shape is (L*D, B, H). | FLOAT32, FLOAT16 | ND |
| r | Output | Denotes the reset gate output, corresponding to $r_t$ in the formula. In fixed-length mode, the shape is (T, B, H); in variable-length mode, the shape is (sum(batch_size), H). | FLOAT32, FLOAT16 | ND |
| z | Output | Denotes the update gate output, corresponding to $z_t$ in the formula. In fixed-length mode, the shape is (T, B, H); in variable-length mode, the shape is (sum(batch_size), H). | FLOAT32, FLOAT16 | ND |
| n | Output | Denotes the new gate output, corresponding to $n_t$ in the formula. In fixed-length mode, the shape is (T, B, H); in variable-length mode, the shape is (sum(batch_size), H). | FLOAT32, FLOAT16 | ND |
| n_h | Output | Denotes the hidden state-new gate output $W_{hn} h_{(t-1)} + b_{hn}$. In fixed-length mode, the shape is (T, B, H); in variable-length mode, the shape is (sum(batch_size), H). | FLOAT32, FLOAT16 | ND |
| direction | Attribute | Denotes the GRU direction. The value "UNIDIRECTIONAL" indicates the unidirectional mode. | STRING | - |
| is_training | Attribute | Denotes whether the mode is training mode. The default value is true. | BOOL | - |

## Constraints

- The input x can have only 2 dimensions (variable-length mode) or 3 dimensions (fixed-length mode).
- The shape of the weight wi is $[3H, I]$ for the first layer and $[3H, D*H]$ for non-first layers.
- The shape of the weight wh is $[3H, H]$.
- The shapes of the biases bi and bh are $[3H]$.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_gru](examples/test_aclnn_gru.cpp) | Invokes the Gru operator through the [aclnnGRU](docs/aclnnGRU_en.md) interface. |
