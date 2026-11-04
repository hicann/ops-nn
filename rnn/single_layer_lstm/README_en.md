# SingleLayerLstm

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

Only the 950 series is supported. The OpDef configuration value is `ascend950`. The internal operator executes a single-layer, unidirectional LSTM; multi-layer, layout conversion, and physical dimension padding are orchestrated by the common `aclnnLSTM`.

The common 950-series path accepts FP32, FP16, and BF16, supports time-first / batch-first, optional bias, multiple layers, and train / eval, and requires unidirectional mode, dropout=0, and non-packed input. Unsupported combinations return an error.

The following text uses 950 series, A3 series, and A2 series as abbreviations for the preceding Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products respectively.

## Function

- Operator function: computes the output sequence of a single-layer, unidirectional LSTM and outputs seven types of saved states for backward use.

The gate order is `ifjo`:

```text
[i_hat, f_hat, j_hat, o_hat] = concat(x_t, h_prev) @ w + b_eff
b_eff = b                         # bias_hh is not passed
b_eff = fp32(b) + fp32(bias_hh)    # when both raw biases are passed, they are added in a private FP32 space
i = sigmoid(i_hat); f = sigmoid(f_hat); j = tanh(j_hat); o = sigmoid(o_hat)
c = f * c_prev + i * j
h = o * tanh(c)
```

The implementation flow is as follows:

```text
aclnnLSTM
  → dynamic_rnn/op_api/lstm_single_layer_adapter.h
  → l0op::SingleLayerLstm
  → op_kernel/arch35/single_layer_lstm.cpp
```

The kernel sequentially performs input projection, full-timeline recurrence, and output scattering. The on-chip layout is shared by host and device; whether weights and states stay resident depends on the budget, and some shapes require access to GM. Reductions with a long K use segmented compensated accumulation, and tiling affects the floating-point addition order.

Backward is completed by the common `aclnnLstmBackward → l0op::SingleLayerLstmGrad`, which consumes saved states of the same declared type.

## Parameters

The following are the parameters of the internal operator `SingleLayerLstm`. For the common interface, see the parameter list of `aclnnLSTM`. `T` is the sequence length, `B` is the batch size, and `I/H` are the physical input/hidden dimensions of internal tensors. The public logical dimensions and padding constraints are described below. FLOAT, FLOAT16, and BFLOAT16 correspond to FP32, FP16, and BF16 respectively.

<table>
  <thead>
    <tr><th>Parameter name</th><th>Input/Output/Attribute</th><th>Description</th><th>Data type</th><th>Data format</th></tr>
  </thead>
  <tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Input sequence with shape <code>[T, B, I]</code>; internally only the time-first layout is used.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>w</td>
    <td>Input</td>
    <td>Fused weight with shape <code>[I+H, 4H]</code>; along rows come the input weights first and then the recurrent weights, and along columns the gates are arranged as <code>ifjo</code>. When SingleLayerLstmGrad uses the same gate order, its <code>[4H, I+H]</code> weight is the transpose of this parameter.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>Input</td>
    <td>Bias with shape <code>[4H]</code>. When <code>bias_hh</code> is not passed, it is the fused bias; when <code>bias_hh</code> is passed, it is the raw input bias <code>bias_ih</code>. The common API provides a zero bias when it has no bias, and this internal input remains mandatory.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_h</td>
    <td>Input</td>
    <td>Initial hidden state with shape <code>[B, H]</code>; different from the <code>[1, B, H]</code> of SingleLayerLstmGrad.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_c</td>
    <td>Input</td>
    <td>Initial cell state with shape <code>[B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seq_length</td>
    <td>Optional input</td>
    <td>Single maximum valid step count (scalar or <code>[1]</code>), applicable to the entire batch; valid values are in <code>[0, T]</code>, outputs with <code>t &gt;= seq_length</code> are set to zero, and the current value handling is described in Constraints.</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias_hh</td>
    <td>Optional input</td>
    <td>Raw recurrent bias with shape <code>[4H]</code>; it is widened separately from <code>b</code> and then added in the private FP32 space of the kernel.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Hidden state at each time step with shape <code>[T, B, H]</code>; the common output sequence is obtained by slicing this output and converting its layout.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_h</td>
    <td>Output</td>
    <td>Saved hidden state at each time step with shape <code>[T, B, H]</code>; its values and dtype are identical to those of <code>y</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>output_c</td>
    <td>Output</td>
    <td>Saved cell state at each time step with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>i</td>
    <td>Output</td>
    <td>Sigmoid activation value of the input gate with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>j</td>
    <td>Output</td>
    <td>Tanh activation value of the candidate cell state with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>f</td>
    <td>Output</td>
    <td>Sigmoid activation value of the forget gate with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>o</td>
    <td>Output</td>
    <td>Sigmoid activation value of the output gate with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tanhc</td>
    <td>Output</td>
    <td><code>tanh(output_c)</code> at each time step with shape <code>[T, B, H]</code>; it corresponds to the input <code>tanhct</code> of SingleLayerLstmGrad.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>direction</td>
    <td>Optional attribute</td>
    <td>Only <code>UNIDIRECTIONAL</code> is supported, and the default value is the same; unsupported values return an error.</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gate_order</td>
    <td>Optional attribute</td>
    <td>Only <code>ifjo</code> (input gate, forget gate, candidate state, output gate) is supported, and the default value is the same.</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>logical_input_size</td>
    <td>Optional attribute</td>
    <td>Number of input features before padding; the default <code>-1</code> means that the physical dimension <code>I</code> is used, and the range of an explicit value is <code>[0, I]</code>. The storage shape keeps the physical dimension.</td>
    <td>INT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>logical_hidden_size</td>
    <td>Optional attribute</td>
    <td>Number of hidden-state features before padding; the default <code>-1</code> means that the physical dimension <code>H</code> is used, and the range of an explicit value is <code>(0, H]</code>. The storage shape keeps the physical dimension.</td>
    <td>INT</td>
    <td>-</td>
  </tr>
  </tbody>
</table>

All floating-point IO uses the same declared type, except for the integer `seq_length`. The kernel widens x, w, and b in a private workspace, performs input projection and recurrence in FP32, and narrows the results when writing the eight outputs. States passed across layers and across forward/backward all use the declared dtype.

The common API passes the two raw biases separately, and the kernel adds them in a private FP32 workspace, avoiding the extra rounding or overflow of converting the fused result back to FP16/BF16. Internal calls that omit bias_hh still use b as the fused bias; the outputs and saved states all use the input dtype.

## Constraints

- The `T/B/I/H` of internal tensors must all be greater than 0, and the physical `I/H` must both be multiples of 8; all floating-point IO uses the same dtype and the ND format. The common API pads the weights per gate and performs the final slicing, and internal outputs still keep the physical shapes.
- The common logical `I=0` still performs a non-empty recurrence, which uses a positive physical input dimension after adaptation and sets `logical_input_size=0`. The common `B=0/H=0` is handled by the adapter layer for empty outputs; the internal kernel requires positive physical `B/H`.
- `output_h/output_c/i/j/f/o/tanhc` are saved states over the full timeline, and their dtype is the same as that of the input. The `h_n/c_n` of the common API are extracted by the adapter layer.
- When `seq_length` is not provided, execution covers the full `T`. The current host implementation reads the first obtainable INT64 value; when no data is obtainable, the input is empty, or the value is not in `[0,T]`, it is also treated as `T`. The common non-packed ACLNN adaptation does not pass this input; this parameter represents a unified time-step length, while the parameter of the same name in Grad is an element-wise mask, so the two cannot be interchanged.
- Internally only `direction=UNIDIRECTIONAL` and `gate_order=ifjo` are supported. The common API is responsible for multiple layers, batch-first conversion, and train/eval orchestration; shapes that do not satisfy the tiling or on-chip resource budget requirements return an error.

## Invocation Description

| Invocation method | Interface document | Description |
| --- | --- | --- |
| aclnn interface | [aclnnLSTM](../dynamic_rnn/docs/aclnnLSTM_en.md) | The 950-series non-packed input path calls SingleLayerLstm through the adapter layer; for common parameters, shapes, and restrictions, the interface document prevails. |
