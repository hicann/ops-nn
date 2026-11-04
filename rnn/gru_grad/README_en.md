# GruGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: **Backward operator** of the gated recurrent unit (GRU), corresponding to the forward operator [Gru](../gru/README.md). Given the forward inputs and intermediate results of each gate, as well as the output gradients from the upper layer, the operator computes the gradients of the input, initial hidden state, weights, and bias. Multi-layer stacking, bidirectional mode, and modes with or without bias are supported. The input is a fixed-length 3D tensor.

- Computation idea (single-layer unidirectional, BPTT, time steps from `T-1` to `0`): Let `hp = init_h (t=0)` or `output_h[t-1]`, and let the gate order be `r, z, n`. Then:

  $$
  \text{grad\_h}_t = \text{dy}[t] + \text{dh}_{\text{next}}
  $$

  $$
  d n = \text{grad\_h}_t \odot (1 - z_t), \quad d z_{\text{raw}} = \text{grad\_h}_t \odot (h_p - n_t), \quad d h_{\text{prev\_from\_h}} = \text{grad\_h}_t \odot z_t
  $$

  $$
  d i_n = d n \odot (1 - n_t^2), \quad d r = (d i_n \odot h_n) \odot r_t \odot (1 - r_t), \quad d z = d z_{\text{raw}} \odot z_t \odot (1 - z_t)
  $$

  $$
  d g_i = [d r,\ d z,\ d i_n], \quad d g_h = [d r,\ d z,\ d i_n \odot r_t]
  $$

  After all time steps are aggregated, the following results are obtained through matmul and reduce (the weights use the kernel layout `w_input[I,3H]` and `w_hidden[H,3H]`):

  $$
  d w_{input} = x^\top \cdot d g_i \quad [I, 3H], \qquad d w_{hidden} = h_p^\top \cdot d g_h \quad [H, 3H]
  $$

  $$
  d x = d g_i \cdot w_{input} \quad [T, B, I], \qquad d h_{prev} = d g_h \cdot w_{hidden} \quad [1, B, H]
  $$

  $$
  d b_{input} = \sum_{T,B} d g_i \quad [3H], \qquad d b_{hidden} = \sum_{T,B} d g_h \quad [3H]
  $$

  Here $\odot$ denotes element-wise multiplication; `dh_next` is the hidden state gradient passed back from the next time step (when `t=T-1`, it is provided by the input `dh`).

- Multi-layer bidirectional mode: The aclnn API `aclnnGRUBackward` orchestrates the process on the host side. It invokes this operator layer by layer (`L-1 → 0`) and direction by direction. `dy` is split along the feature dimension by direction, `dx` is summed over the two directions and chained back as the `dy` of the lower layer, and `dh_prev`/`dw`/`db` are routed by layer/direction.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
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
      <td>Forward input sequence, corresponding to $x_t$ in the formula. The fixed-length shape is (T, B, I).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>w_input</td>
      <td>Input</td>
      <td>Input-side weight (kernel layout), corresponding to $W_{ir}/W_{iz}/W_{in}$. The shape is $[3H, I]$ (first layer) or $[3H, D*H]$ (non-first layer).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_hidden</td>
      <td>Input</td>
      <td>Hidden-state-side weight (kernel layout), corresponding to $W_{hr}/W_{hz}/W_{hn}$. The shape is $[3H, H]$.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>init_h</td>
      <td>Input</td>
      <td>Initial hidden state $h_0$. The shape is $[1, B, H]$ (single-layer unidirectional).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>output_h</td>
      <td>Input</td>
      <td>Hidden state output $h_t$ of each forward step. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>reset_gate</td>
      <td>Input</td>
      <td>Forward reset gate activation value $r_t$. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>update_gate</td>
      <td>Input</td>
      <td>Forward update gate activation value $z_t$. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>new_gate</td>
      <td>Input</td>
      <td>Forward new gate activation value $n_t$. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>h_n</td>
      <td>Input</td>
      <td>Forward pre-activation of the new gate on the hidden state side, $W_{hn}h_{t-1}+b_{hn}$. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Output gradient of each time step. The shape is (T, B, H).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>dh</td>
      <td>Input</td>
      <td>Hidden state gradient at the last time step (from the next time step or the upper layer). The shape is [1, B, H].</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>batch_sizes</td>
      <td>Optional input</td>
      <td>Number of valid batches at each moment of variable-length sequences. The shape is (T).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>Output</td>
      <td>Gradient of input x. The shape is the same as that of x: (T, B, I).</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>dh_prev</td>
      <td>Output</td>
      <td>Gradient of the initial hidden state $d h_0$. The shape is [1, B, H].</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND, NCL</td>
    </tr>
    <tr>
      <td>dw_input</td>
      <td>Output</td>
      <td>$d w_{input}$. The shape is the same as that of w_input.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dw_hidden</td>
      <td>Output</td>
      <td>$d w_{hidden}$. The shape is the same as that of w_hidden.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db_input</td>
      <td>Output</td>
      <td>$d b_{input}$. The shape is $[3H]$. This output is not computed when has_bias=false.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>db_hidden</td>
      <td>Output</td>
      <td>$d b_{hidden}$. The shape is $[3H]$. This output is not computed when has_bias=false.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>direction</td>
      <td>Attribute</td>
      <td>GRU direction. Value: "UNIDIRECTIONAL".</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>has_bias</td>
      <td>Attribute</td>
      <td>Indicates whether to compute bias gradients. When the value is false, the kernel skips bias gradient computation and does not output a valid db. Default value: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Attribute</td>
      <td>Indicates whether the mode is training mode (the backward operator is always true, aligned with the forward operator). Default value: true.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batch_first</td>
      <td>Attribute</td>
      <td>Indicates whether the batch of input/dy/dx is in the first dimension. Default value: false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Input x is a 3D fixed-length tensor of (T, B, I).
- The weights use the kernel layout: `w_input` is $[3H, I]$ (first layer) or $[3H, D*H]$ (non-first layer), and `w_hidden` is $[3H, H]$. The `params` input of the aclnn API is passed in the PyTorch layout $[3H, *]$.
- The gate inputs (reset_gate/update_gate/new_gate/h_n/output_h/dy) must all be of (T, B, H), and their dtype must be the same as that of x.
- When `has_bias=false`, bias gradients are not computed (`db_input`/`db_hidden` have no valid outputs).
- x/dy/dh/output_h and the gate inputs support both the ND and NCL formats; `batch_sizes` (1D) supports only ND.
