# DynamicAUGRU

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: DynamicAUGRU implements the attention-based update gate GRU (a gated recurrent unit whose update gate is modulated by attention scores) and processes sequence data such as user behaviors. The operator updates the hidden state at each time step and returns the complete hidden state sequence and intermediate gating results.

- Input and output shapes: Let `T` denote the number of time steps, `B` the batch size, `I` the input feature dimensions, and `H` the hidden state dimensions. The shape of input `x` is `[T, B, I]`, and the shapes of the seven outputs are all `[T, B, H]`.

- Formula: The time index starts from 0, and $h_{-1}$ is `init_h[0]`; if the initial state is not provided, it is set to zero. The input projection and the hidden state projection at each time step are:

  $$
  G^x_t = x_t W_x + b_x, \qquad G^h_t = h_{t-1} W_h + b_h
  $$

  Here, $W_x$ and $W_h$ correspond to `weight_input` and `weight_hidden` respectively, and $b_x$ and $b_h$ correspond to `bias_input` and `bias_hidden` respectively; biases that are not provided are treated as zero. The last dimension of the projections is divided into three segments: the update gate, the reset gate, and the candidate state, with subscripts $z$, $r$, and $n$ respectively. When `gate_order="zrh"`, the storage order is $z,r,n$; when `gate_order="rzh"`, the storage order is $r,z,n$.

  $$
  \begin{aligned}
  z_t &= \sigma(G^x_{t,z} + G^h_{t,z}) \\
  r_t &= \sigma(G^x_{t,r} + G^h_{t,r}) \\
  n_t &= \tanh(G^x_{t,n} + r_t \odot G^h_{t,n}) \\
  \widehat{z}_t &= (1-a_t) \odot z_t \\
  \widetilde{h}_t &= n_t + \widehat{z}_t \odot (h_{t-1}-n_t)
  \end{aligned}
  $$

  $\sigma$ is the Sigmoid function, $\odot$ denotes element-wise multiplication, and $a_t$ is `weight_att[t]`, which is broadcast along the hidden dimension.

## Parameters

All tensors use the ND format. FLOAT in the table denotes a 32-bit floating-point number.

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 470px">
  <col style="width: 170px">
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
      <td>x</td>
      <td>Input</td>
      <td>Input sequence features, corresponding to x<sub>t</sub> in the formula, with shape [T, B, I].</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_input</td>
      <td>Input</td>
      <td>Input projection weight, corresponding to W<sub>x</sub> in the formula, with shape [I, 3H].</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_hidden</td>
      <td>Input</td>
      <td>Hidden state projection weight, corresponding to W<sub>h</sub> in the formula, with shape [H, 3H].</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight_att</td>
      <td>Input</td>
      <td>Attention scores, corresponding to a<sub>t</sub> in the formula, with shape [T, B].</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias_input</td>
      <td>Optional input</td>
      <td>Input projection bias, corresponding to b<sub>x</sub> in the formula, with shape [3H]; if not provided, it is treated as zero.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias_hidden</td>
      <td>Optional input</td>
      <td>Hidden state projection bias, corresponding to b<sub>h</sub> in the formula, with shape [3H]; if not provided, it is treated as zero.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_length</td>
      <td>Optional input</td>
      <td>Sequence control input. When the type is INT32, the shape is [B]; when the type is FLOAT16, it serves as an element-wise mask with shape [T, B, H]; if not provided, no sequence control is performed.</td>
      <td>INT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>Optional input</td>
      <td>Initial hidden state, with shape [1, B, H]; if not provided, it is treated as zero.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Hidden state sequence after sequence control is applied, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>Output</td>
      <td>Complete hidden state sequence, the same as y, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update</td>
      <td>Output</td>
      <td>Update gate z<sub>t</sub>, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_att</td>
      <td>Output</td>
      <td>Update gate z&#770;<sub>t</sub> modulated by attention, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reset</td>
      <td>Output</td>
      <td>Reset gate r<sub>t</sub>, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>new</td>
      <td>Output</td>
      <td>Candidate hidden state n<sub>t</sub>, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>hidden_new</td>
      <td>Output</td>
      <td>Hidden projection G<sup>h</sup><sub>t,n</sub> corresponding to the candidate state, which includes the hidden-side bias and has not yet been multiplied by the reset gate, with shape [T, B, H].</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>Optional attribute</td>
      <td>Computation direction. The default value is "UNIDIRECTIONAL", and only this value is supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_depth</td>
      <td>Optional attribute</td>
      <td>Number of recurrent cell layers. The default value is 1, and only this value is supported.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keep_prob</td>
      <td>Optional attribute</td>
      <td>Keep probability. The default value is 1.0, only this value is supported, and Dropout is not performed.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cell_clip</td>
      <td>Optional attribute</td>
      <td>State clipping parameter. The default value is -1.0, only this value is supported, and clipping is not performed.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_proj</td>
      <td>Optional attribute</td>
      <td>Projection dimension reduction parameter. The default value is 0, only this value is supported, and projection dimension reduction is not performed.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>time_major</td>
      <td>Optional attribute</td>
      <td>Whether the time dimension is placed first. The default value is true, and only this value is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activation</td>
      <td>Optional attribute</td>
      <td>Activation function for the candidate state. The default value is "tanh", and only this value is supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gate_order</td>
      <td>Optional attribute</td>
      <td>Gate arrangement order of weights and biases. The default value is "zrh", and "zrh" and "rzh" are supported.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reset_after</td>
      <td>Optional attribute</td>
      <td>Whether the reset gate is applied after the hidden projection and the bias are added. The default value is true, and only this value is supported.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Optional attribute</td>
      <td>Training mode flag. The default value is true, and both true and false are supported; both values return seven outputs.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Only single-layer, unidirectional forward computation with the time dimension placed first is supported.
- At runtime, `T`, `B`, `I`, and `H` must all be greater than 0; empty tensors are not supported. The associated dimensions of each input must satisfy the shape relationships in the Parameters section.
- At invocation, inputs are organized in the contiguous ND layout; this directory does not provide dedicated invocation examples or verification results for non-contiguous inputs.
- The provided tensors among `bias_input`, `bias_hidden`, and `init_h` must use the same data type, and the seven outputs are consistent with them. When none of the three is provided, the output type is FLOAT16.
- When `seq_length` is not provided, `y` and `output_h` store the hidden state computed at the current time step.
- When `seq_length` is an INT32 tensor of shape `[B]`, for time step `t` of sample `b`:
  - When `t < seq_length[b]`, `y[t, b, :]` and `output_h[t, b, :]` store the hidden state computed at the current time step.
  - When `t >= seq_length[b]`, the hidden state of the previous time step is reused; when `t` is 0, `init_h[0, b, :]` is used, and when `init_h` is not provided, an all-zero state is used.
  - When `seq_length[b]` is less than or equal to 0, the initial state is kept throughout; when it is greater than or equal to `T`, all time steps participate in the computation.
- When `seq_length` is a FLOAT16 tensor of shape `[T, B, H]`, `seq_length[t, b, h]` serves as the blending coefficient of the corresponding hidden state element. The value 0 reuses the hidden state of the previous time step, the value 1 uses the hidden state computed at the current time step, and other values are processed by linear blending.
- Sequence control affects only `y`, `output_h`, and the recurrent states of subsequent time steps; `update`, `update_att`, `reset`, `new`, and `hidden_new` still store the computation results of the current time step.
- Dynamic shape and dynamic rank are supported at the compilation stage; at runtime, they must resolve to concrete shapes that satisfy the constraints.
- Internal state recurrence uses FP32 computation, and outputs are written back in the declared data types; rounded output values are not fed back into the recurrence. When the sequence is split into multiple invocations, bitwise identity with a single complete invocation is not guaranteed.

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| Graph-mode invocation | [test_geir_dynamic_augru](examples/arch35/test_geir_dynamic_augru.cpp) | Invokes DynamicAUGRU through graph construction with the [operator IR](op_graph/dynamic_augru_proto.h), and verifies static shape, dynamic shape, and dynamic rank. |

The `OriginOpType` registered by the TF plugin is `DynamicAUGRU`. This registration maps TF graph nodes to GE operators; you cannot infer from it that a `tf.raw_ops` with the same name or a TF-Adapter Python function exists. This directory provides GEIR invocation examples and does not provide TF/ONNX end-to-end import examples. No dedicated verification result is available for the bitwise determinism of repeated executions.
