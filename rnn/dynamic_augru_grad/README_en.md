# DynamicAUGRUGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                            | Support |
| :---------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>         |    √     |
| <term>Atlas A3 products</term>                   |    ×     |
| <term>Atlas A2 products</term>                   |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>          |    ×     |
| <term>Atlas inference products</term>                  |    ×     |
| <term>Atlas training products</term>                  |    ×     |

## Function

- Operator function: backward operator of the GRU with an attentional update gate (**AUGRU**). Given the forward input, weights, attention, intermediate results of each gate, and gradients from the upper layer, it computes the gradients of the input, initial hidden state, weights, and biases according to BPTT (from time step `T-1` to `0`), and additionally outputs the attention gradient `dw_att`. When `seq_length` is passed, the kernel generates the variable-length mask online (`seq_mask[t,b] = (t < seq_length[b]) ? 1 : 0`); when it is not passed, this is equivalent to an all-1 mask.

- Formula (single-layer unidirectional, with gate_order set to zrh; for rzh the z/r slots are swapped). Let $u_t = z_t \cdot (1 - \text{att}_t)$, processed in reverse time-step order:

  $$
  \text{grad\_h}_t = \text{mask}_t \odot \text{dh}_{\text{prev}} + \text{dy}[t]
  $$

  $$
  \text{dnt} = \text{grad\_h}_t \odot (1 - u_t) \odot (1 - n_t^2), \qquad \text{dr} = \text{dnt} \odot r_t \odot (1 - r_t) \odot \tilde{n}_t
  $$

  $$
  \text{dz} = \text{grad\_h}_t \odot (h_{t-1} - n_t) \odot (1 - \text{att}_t) \odot z_t \odot (1 - z_t), \qquad \text{dw\_att}_t = -\sum_h (\text{grad\_h}_t \odot (h_{t-1} - n_t) \odot z_t)
  $$

  $$
  \text{dh}_{\text{prev}} = [\text{dz}, \text{dr}, \text{dnt} \odot r_t] \cdot W_{hh}^T + \text{grad\_h}_t \odot u_t
  $$

  After the loop finishes, aggregation is performed (the weights use the kernel layouts `w_input[I,3H]` and `w_hidden[H,3H]`):

  $$
  d w_{input} = x^T \cdot [\text{dz}, \text{dr}, \text{dnt}], \quad d w_{hidden} = h_{prev}^T \cdot [\text{dz}, \text{dr}, \text{dnt} \odot r_t], \quad d x = [\text{dz}, \text{dr}, \text{dnt}] \cdot W_{in}^T
  $$

  $$
  d b_{input} = \sum_{T,B} [\text{dz}, \text{dr}, \text{dnt}], \qquad d b_{hidden} = \sum_{T,B} [\text{dz}, \text{dr}, \text{dnt} \odot r_t]
  $$

## Parameters

In the following table, `T` denotes the sequence length, `B` denotes the batch size, `I` denotes the input feature dimensions, and `H` denotes the hidden state dimensions.

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ------ | -------------- | ---- | -------- | -------- |
| x | Input | Forward input sequence with shape `[T, B, I]`. | FLOAT32, FLOAT16 | ND |
| weight_input | Input | Input-side weight with shape `[I, 3H]`. | FLOAT32, FLOAT16 | ND |
| weight_hidden | Input | Hidden-state-side weight with shape `[H, 3H]` or `[1, H, 3H]`. Both shapes use the same contiguous `H * 3H` layout. | FLOAT32, FLOAT16 | ND |
| weight_att | Input | Attention scores already broadcast along the hidden state dimension, with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| y | Input | Forward output with shape `[T, B, H]`. The current implementation treats it as a placeholder input, and it does not participate in numerical computation. | FLOAT32, FLOAT16 | ND |
| init_h | Input | Initial hidden state with shape `[B, H]`. | FLOAT32, FLOAT16 | ND |
| h | Input | Hidden state outputs of the forward at each time step, with shape `[T, B, H]`, where `h[t]` is the output of time step `t`. | FLOAT32, FLOAT16 | ND |
| dy | Input | Gradients of the outputs at each time step, with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| dh | Input | Gradient of the hidden state at the last time step, with shape `[B, H]`. | FLOAT32, FLOAT16 | ND |
| update | Input | Forward update gate activation value z<sub>t</sub> with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| update_att | Input | Update gate after the attention effect, u<sub>t</sub> = z<sub>t</sub> · (1 - att<sub>t</sub>), with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| reset | Input | Forward reset gate activation value r<sub>t</sub> with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| new | Input | Forward new gate activation value n<sub>t</sub> with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| hidden_new | Input | Hidden-state-side pre-activation value of the new gate, with shape `[T, B, H]`. | FLOAT32, FLOAT16 | ND |
| seq_length | Optional input | Actual sequence length of each batch, with shape `[B]`. When this parameter is passed, the operator generates the mask internally; when it is not passed, this is equivalent to an all-1 mask. | INT32 | ND |
| mask | Optional input | Placeholder reserved for dropout; the current implementation does not use it in numerical computation. | UINT8 | ND |
| dw_input | Output | Gradient of the input-side weight, with shape `[I, 3H]`. | FLOAT32, FLOAT16 | ND |
| dw_hidden | Output | Gradient of the hidden-state-side weight, with shape `[H, 3H]`. | FLOAT32, FLOAT16 | ND |
| db_input | Output | Gradient of the input-side bias, with shape `[3H]`. | FLOAT32, FLOAT16 | ND |
| db_hidden | Output | Gradient of the hidden-state-side bias, with shape `[3H]`. | FLOAT32, FLOAT16 | ND |
| dx | Output | Gradient of the forward input sequence, with shape `[T, B, I]`. | FLOAT32, FLOAT16 | ND |
| dh_prev | Output | Gradient of the initial hidden state, with shape `[B, H]`. | FLOAT32, FLOAT16 | ND |
| dw_att | Output | Gradient of the attention, with shape `[T, B]`. | FLOAT32, FLOAT16 | ND |
| direction | Optional attribute | Loop direction. The default value is `"UNIDIRECTIONAL"`, and currently only the default value is supported. | STRING | - |
| cell_depth | Optional attribute | Number of loop layers. The default value is `1`, and currently only the default value is supported. | INT | - |
| keep_prob | Optional attribute | Dropout retention probability. The default value is `-1.0`; currently `1.0` and `-1.0` are supported, and dropout does not take effect. | FLOAT | - |
| cell_clip | Optional attribute | Cell clipping threshold. The default value is `-1.0`, and currently only the default value is supported, so cell clipping does not take effect. | FLOAT | - |
| num_proj | Optional attribute | Projection dimensions. The default value is `0`, and currently only the default value is supported, which means that projection is not enabled. | INT | - |
| time_major | Optional attribute | Whether the input uses a time-major layout. The default value is `true`, and currently only the default value is supported. | BOOL | - |
| gate_order | Optional attribute | Arrangement order of the gates in the weights and gradients. The default value is `"zrh"`; `"zrh"` and `"rzh"` are supported. | STRING | - |
| reset_after | Optional attribute | Whether the reset gate is applied after the matrix multiplication. The default value is `true`, and currently only the default value is supported. | BOOL | - |

## Constraints

- x is a 3D fixed-length tensor of [T, B, I]; at runtime T, B, I, and H must all be greater than 0, and tensors with empty computation dimensions are not supported. When H or I is not aligned to 16, the padded layout is used, and the pad region is always 0.
- The inputs declare automatic contiguous conversion (`AutoContiguous`), and the kernel consumes contiguous ND layouts; the graph-path behavior for non-contiguous inputs has no dedicated field-test results yet.
- FLOAT32 and FLOAT16 are supported; both dtypes reuse the same fp32 cube path (fp16 performs Cast only at the input/output boundaries, and precision is based on fp32 accumulation).
- Except for seq_length (INT32) and mask (UINT8), all floating-point inputs/outputs have the same dtype as x.
- Elements of `seq_length` satisfy 0 ≤ seq_length[b] ≤ T; out-of-range values are not additionally validated, and the kernel naturally clamps them by the `t < seq_length[b]` comparison (greater than T is equivalent to T, and less than 0 is equivalent to 0).
- Bidirectional, multi-layer (cell_depth>1), and projection (num_proj>0) modes are not supported yet.
- `keep_prob` accepts only 1.0/-1.0 (dropout does not take effect), and `cell_clip` accepts only -1.0 (clip does not take effect); other values report an error.
- The dynamic shape sample declares T, B, and I as `-1` and keeps H as a concrete value; fully dynamic H is not supported. The dynamic rank sample declares only the optional input `mask`, which does not participate in computation, as `[-2]`; you cannot infer from this that the core computation inputs support unknown ranks.

## Invocation Description

The operator is invoked in GE Graph mode. Released binaries are generated automatically by the build flow, and the source directory does not provide handwritten binary JSON configurations:

| Invocation method | Invocation sample | Description |
| -------- | -------- | ---- |
| GE graph mode | [test_geir_dynamic_augru_grad](examples/arch35/test_geir_dynamic_augru_grad.cpp) | Builds the graph through the [operator IR](op_graph/dynamic_augru_grad_proto.h) and invokes the DynamicAUGRUGrad operator. |
| GE graph mode | [test_geir_dynamic_augru_grad_dynamic](examples/arch35/test_geir_dynamic_augru_grad_dynamic.cpp) | Verifies unknown T/B/I dimensions in the same graph; it also uses the optional placeholder input mask to verify unknown ranks, while the core computation inputs still use known ranks. |

The `OriginOpType` registered by the TF plugin is `DynamicAUGRUGrad`. This registration is used for mapping TF graph nodes to GE operators, and you cannot infer from it that a same-named `tf.raw_ops` or TF-Adapter Python function exists. This directory provides a GEIR invocation sample and does not provide a TF/ONNX end-to-end import sample. There are no dedicated verification results for bitwise determinism across repeated executions.
