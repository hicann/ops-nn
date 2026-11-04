# SingleLayerLstmGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                                            | Support |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR&950DT products</term>                                                |    √    |
| <term>Atlas A3 products</term>                          |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                                          |    ×     |
| <term>Atlas inference products</term>                                                 |    ×     |
| <term>Atlas training products</term>                                                  |    ×     |

The following text uses 950 series, A3 series, and A2 series as shorthand for the Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products, respectively.

## Function

**Architecture and type configuration:**

- A2 series: The OpDef configuration value is `ascend910b`; the operator supports FP32/FP16 and uses `op_kernel/single_layer_lstm_grad.cpp`.
- A3 series: The OpDef configuration value is `ascend910_93`; the operator supports FP32/FP16 and uses `op_kernel/single_layer_lstm_grad.cpp`.
- 950 series: The OpDef configuration value is `ascend950`; the operator registers FP32/FP16/BF16 through a separate `OpAICoreConfig` and uses `op_kernel/arch35/single_layer_lstm_grad.cpp`. All floating-point inputs, saved states, and gradient outputs use the same declared type. The kernel widens inputs to FP32 internally, accumulates in FP32, and then narrows the outputs; caches shared across operators use the declared type.
- The regbase path of the 950 series uses `DTYPE_W`. For the narrow-type large-shape path, the operator first widens inputs in a private workspace and recomputes them, then invokes the FP32 matmul/backward path, and finally narrows the outputs.

The FP16/BF16 backward of the 950 series recomputes FP32 gate values and states from `x/w/init_h/init_c/b` inside the kernel and then performs the backward recurrence, which reduces the impact of low-precision cache rounding on gradient computation. The seven types of external caches and gradients remain in the declared dtype; FP32 recompute results are used only within the current kernel. The small-shape path processes batch blocks independently on AIVs; the large-shape path uses a private workspace and synchronizes hidden states generated across cores between time steps.

For narrow types, `b` supports the original fused bias `[4H]` and also supports `[8H]` concatenated in the order of `bias_ih` and `bias_hh`. The common ACLNN adaptation uses the latter and adds the two parts after widening inside the kernel; `db` is always `[4H]`, and the adaptation layer maps the same derivative back to the two biases separately. The FP32 path and the A2/A3 series use the fused bias `[4H]`.

- Operator function: Backward propagation of a single-layer unidirectional LSTM. The operator computes the gradients of the forward input x, weight w, bias b, initial hidden state initH, and initial cell state initC.

- Formula:

Single-layer LSTM backward computation

**Forward propagation formulas**

| Component | Formula |
|------|------|
| Input concatenation | $\mathbf{z}_t = \begin{bmatrix} \mathbf{x}_t \\ \mathbf{h}_{t-1} \end{bmatrix}$ |
| Forget gate | $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{b}_f)$ |
| Input gate | $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{b}_i)$ |
| Candidate state | $\mathbf{g}_t = \tanh(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_g)$ |
| Output gate | $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{b}_o)$ |
| Cell state | $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ |
| Hidden state | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

Where:

- $\sigma$ is the sigmoid function
- $\odot$ denotes element-wise multiplication (Hadamard product)
- $W_*$ is a learnable weight matrix
- $b_*$ is a learnable bias term

**Backward propagation variable definitions**

- Total loss: the scalar $L$; the caller provides the upstream gradients `dy`, `dh`, and `dc` for the output sequence, the final hidden state, and the final cell state, respectively.
- Hidden state gradient: $\delta\mathbf{h}_t = \frac{\partial L}{\partial \mathbf{h}_t}$
- Cell state gradient: $\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{c}_t}$

**Backward propagation algorithm (time step $t \rightarrow t-1$)**

**Initialization**

$$
\delta\mathbf{h}_{\text{next}} = \mathrm{dh}, \quad \delta\mathbf{c}_{\text{next}} = \mathrm{dc}
$$

The following example uses a single batch sample with `direction=UNIDIRECTIONAL`; vectors are column vectors, and time steps are numbered from 0. `init_h/init_c` provides the states before step 0. The formulas omit the first dimension of shape 1 from `dh/dc`. Weight and bias gradients accumulate from zero and are summed over all batch samples.

For each time step in the order of $t = T - 1, T - 2, \ldots, 0$, perform the following steps:

1. **Current hidden state gradient**

   $$
   \delta\mathbf{h}_t = \mathrm{dy}_t + \delta\mathbf{h}_{\text{next}}
   $$
2. **Current cell state gradient**

   $$
   \delta\mathbf{c}_t = \delta\mathbf{h}_t \odot \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t)) + \delta\mathbf{c}_{\text{next}}
   $$
3. **Gate gradient computation**

   $$
   \delta\mathbf{o}_t = \delta\mathbf{h}_t \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t \odot (1 - \mathbf{o}_t)
   $$

   $$
   \delta\mathbf{g}_t = \delta\mathbf{c}_t \odot \mathbf{i}_t \odot (1 - \mathbf{g}_t^2)
   $$

   $$
   \delta\mathbf{i}_t = \delta\mathbf{c}_t \odot \mathbf{g}_t \odot \mathbf{i}_t \odot (1 - \mathbf{i}_t)
   $$

   $$
   \delta\mathbf{f}_t = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
   $$
4. **Parameter gradient accumulation**

   $$
   \frac{\partial L}{\partial \mathbf{W}_f} \mathrel{+}= \delta\mathbf{f}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_f} \mathrel{+}= \delta\mathbf{f}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_i} \mathrel{+}= \delta\mathbf{i}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_i} \mathrel{+}= \delta\mathbf{i}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_g} \mathrel{+}= \delta\mathbf{g}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_g} \mathrel{+}= \delta\mathbf{g}_t
   $$

   $$
   \frac{\partial L}{\partial \mathbf{W}_o} \mathrel{+}= \delta\mathbf{o}_t \mathbf{z}_t^\top
   $$

   $$
   \frac{\partial L}{\partial \mathbf{b}_o} \mathrel{+}= \delta\mathbf{o}_t
   $$
5. **Propagation to the previous time step**

   $$

   \delta\mathbf{z}_t = \mathbf{W}_f^\top \delta\mathbf{f}_t + \mathbf{W}_i^\top \delta\mathbf{i}_t + \mathbf{W}_g^\top \delta\mathbf{g}_t + \mathbf{W}_o^\top \delta\mathbf{o}_t
   $$

   $$
   \delta\mathbf{x}_t = \delta\mathbf{z}_t[0:I], \quad
   \delta\mathbf{h}_{\text{prev}} = \delta\mathbf{z}_t[I:I+H]
   $$

   $$
   \delta\mathbf{c}_{\text{prev}} = \delta\mathbf{c}_t \odot \mathbf{f}_t
   $$
6. **Update propagation variables**

   $$
   \delta\mathbf{h}_{\text{next}} \leftarrow \delta\mathbf{h}_{\text{prev}}
   $$

   $$
   \delta\mathbf{c}_{\text{next}} \leftarrow \delta\mathbf{c}_{\text{prev}}
   $$

**Gradient computation principles**

**Derivation of the cell state gradient**

$$
\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} + \frac{\partial L}{\partial \mathbf{c}_{t+1}} \frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t}
$$

Where:

$$
\frac{\partial \mathbf{h}_t}{\partial \mathbf{c}_t} = \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t))
$$

$$
\frac{\partial \mathbf{c}_{t+1}}{\partial \mathbf{c}_t} = \mathbf{f}_{t+1}
$$

**Derivation of the forget gate gradient**

$$
\delta\mathbf{f}_t = \frac{\partial L}{\partial \mathbf{a}_f^t} = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
$$

**Derivation of parameter gradients**

$$
\frac{\partial L}{\partial \mathbf{W}_f} = \sum_{t=0}^{T-1} \delta\mathbf{f}_t \mathbf{z}_t^\top
$$

**LSTM gradient flow characteristics**
**Gradient of the direct cell state connection (with fixed gate values, excluding other paths through hidden state feedback)**

$$
\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_s} = \prod_{k=s+1}^{t} \operatorname{diag}(\mathbf{f}_k), \quad 0 \le s < t < T
$$

**Parameters:**

The following lists the parameters of the internal l0op operator `SingleLayerLstmGrad`. For the common interface, see the parameter list of `aclnnLstmBackward`.
`T` is the sequence length, `B` is the batch size, `I` is the number of input features, and `H` is the number of hidden state features.

All floating-point inputs, saved states, and gradient outputs use the same dtype.

<table>
  <thead>
    <tr><th>Parameter name</th><th>Input/Output/Attribute</th><th>Description</th><th>Data type</th><th>Data format</th></tr>
  </thead>
  <tbody>
  <tr>
    <td>x</td>
    <td>Input</td>
    <td>Input sequence, with shape <code>[T, B, I]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>w</td>
    <td>Input</td>
    <td>Fused weight, with shape <code>[4H, I+H]</code>; along columns, the input weight <code>W_ih</code> comes first and the recurrent weight <code>W_hh</code> follows, and the gate order along rows is specified by <code>gate_order</code>. With the <code>ifjo</code> gate order, the weight is the transpose of the SingleLayerLstm forward weight; with <code>ijfo</code>, the row blocks corresponding to the candidate state and forget gate also need to be swapped.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>b</td>
    <td>Optional Input</td>
    <td>Fused bias, usually <code>[4H]</code>. FLOAT16/BFLOAT16 of the 950 series also supports <code>[8H]</code>: it stores the two original <code>bias_ih</code> and <code>bias_hh</code> in sequence, which are added after widening inside the kernel. Omission means no bias.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Optional Input</td>
    <td>Main forward output, with shape <code>[T, B, H]</code>. The current common <code>aclnnLstmBackward</code> adaptation passes an empty tensor; the hidden state is provided by <code>h</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_h</td>
    <td>Input</td>
    <td>Forward initial hidden state, with shape <code>[1, B, H]</code>; this differs from the <code>[B, H]</code> used for internal input in SingleLayerLstm forward.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>init_c</td>
    <td>Input</td>
    <td>Forward initial cell state, with shape <code>[1, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>h</td>
    <td>Input</td>
    <td>Hidden states saved for each time step in the forward pass, with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>c</td>
    <td>Input</td>
    <td>Cell states saved for each time step in the forward pass, with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dy</td>
    <td>Input</td>
    <td>Upstream gradient for the entire forward output sequence, with shape <code>[T, B, H]</code>. It is an independent input from <code>dh</code>; the two are accumulated at the corresponding time steps.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dh</td>
    <td>Input</td>
    <td>Upstream gradient for the final hidden state of this direction, with shape <code>[1, B, H]</code>. If you do not need this gradient, pass a zero tensor of the same shape; it cannot be omitted.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dc</td>
    <td>Input</td>
    <td>Upstream gradient for the final cell state of this direction, with shape <code>[1, B, H]</code>. If you do not need this gradient, pass a zero tensor of the same shape; it cannot be omitted.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>i</td>
    <td>Input</td>
    <td>Input gate sigmoid activation values saved in the forward pass, with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>j</td>
    <td>Input</td>
    <td>Candidate cell state tanh activation values saved in the forward pass (g in the formulas), with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>f</td>
    <td>Input</td>
    <td>Forget gate sigmoid activation values saved in the forward pass, with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>o</td>
    <td>Input</td>
    <td>Output gate sigmoid activation values saved in the forward pass, with shape <code>[T, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>tanhct</td>
    <td>Input</td>
    <td><code>tanh(c)</code> saved in the forward pass, with shape <code>[T, B, H]</code>; it corresponds to the SingleLayerLstm output <code>tanhc</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>seq_length</td>
    <td>Optional Input</td>
    <td>Per-time-step valid position mask, with shape <code>[T, B, H]</code>; valid positions are 1 and invalid positions are 0. The FLOAT16/BFLOAT16 path of the 950 series does not support passing this mask; for common non-packed calls, pass an empty tensor.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dw</td>
    <td>Output</td>
    <td>Fused weight gradient, with shape <code>[4H, I+H]</code>; the layout and gate order are the same as those of <code>w</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>db</td>
    <td>Output</td>
    <td>Fused bias gradient when a bias exists, with shape <code>[4H]</code>. Even if <code>b</code> is <code>[8H]</code>, only one <code>[4H]</code> output is produced; the derivatives of the two original biases are identical, and the common adaptation returns them separately. When no bias exists, the l0op allocates a placeholder output; it is not read as a valid bias gradient.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dx</td>
    <td>Output</td>
    <td>Input sequence gradient, with shape <code>[T, B, I]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dh_prev</td>
    <td>Output</td>
    <td>Gradient of the initial hidden state <code>init_h</code>, with shape <code>[1, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dc_prev</td>
    <td>Output</td>
    <td>Gradient of the initial cell state <code>init_c</code>, with shape <code>[1, B, H]</code>.</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>direction</td>
    <td>Optional Attribute</td>
    <td>Specifies the traversal direction along the sequence of the corresponding forward layer. Supported values are <code>UNIDIRECTIONAL</code> (default) and <code>REDIRECTIONAL</code>. The latter denotes a unidirectional layer that traverses in reverse; it does not mean that this operator computes a bidirectional LSTM at once, nor does it select whether to compute gradients.</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gate_order</td>
    <td>Optional Attribute</td>
    <td>Gate arrangement of the fused weight and bias. Supported values are <code>ijfo</code> (default) or <code>ifjo</code>; the common <code>aclnnLstmBackward</code> adaptation passes <code>ifjo</code>.</td>
    <td>STRING</td>
    <td>-</td>
  </tr>
  </tbody>
</table>

## Constraints

- The 950 series supports FLOAT (FP32), FLOAT16 (FP16), and BFLOAT16 (BF16); the A2/A3 series supports FLOAT and FLOAT16.
- This operator processes single-layer, unidirectional, time-first internal tensors. Multi-layer handling, batch-first conversion, and gradient splitting per layer/direction are the responsibility of the common ACLNN adaptation; when you call this operator, you must provide internal shapes according to the parameter table.
- `x/w/init_h/init_c`, the three upstream gradients, and the seven types of saved states must come from the same forward configuration. The narrow-type path of the 950 series recomputes states, but the saved state inputs are still required, and their dtypes must match the inputs.
- Internally, the requirements are `T > 0, B > 0, H > 0`. The regbase path of the 950 series supports recurrence with `I = 0`; the shared matmul path requires `I > 0`. The behavior of the common API with `H = 0` is handled by the adaptation layer.
- The FLOAT16/BFLOAT16 of the 950 series does not support the `seq_length` mask; this parameter requires an element-wise mask, which differs from the INT64 length scalar used in the forward pass. The dual-bias `[8H]` form applies only to the 950 series.
- Whether tiling can be generated also depends on the shape, on-chip space, and workspace size; unsupported combinations return an error.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_single_layer_lstm_grad.cpp](examples/test_aclnn_single_layer_lstm_grad.cpp) | Invokes the SingleLayerLstmGrad operator through the [aclnnLstmBackward](docs/aclnnLstmBackward_en.md) interface. |
