# aclnnLstmBackward

## Supported Products

| Product                                                                           | Supported|
| :------------------------------------------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                                               |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>                         |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                                         |    ×     |
| <term>Atlas inference products</term>                                                |    ×     |
| <term>Atlas training products</term>                                                 |    ×     |

## Function

- Function: performs LSTM backpropagation and calculates the gradients of the forward input, weight parameters, and initial state hx.
- Formula:

  <details>

    <summary>Formula for calculating the single-layer LSTM backpropagation</summary>

    | Component| Formula|
    |:---|:---|
    | Input concatenation| $\mathbf{z}_t = \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix}$ |
    | Forget gate| $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{z}_t + \mathbf{b}_f)$ |
    | Input gate| $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{z}_t + \mathbf{b}_i)$ |
    | Candidate state| $\mathbf{g}_t = \tanh(\mathbf{W}_g \mathbf{z}_t + \mathbf{b}_c)$ |
    | Output gate| $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{z}_t + \mathbf{b}_o)$ |
    | Cell state| $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t$ |
    | Hidden state| $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ |

    Where:

    - $\sigma$ is the sigmoid function.
    - $\odot$ indicates element-wise multiplication (Hadamard product).
    - $W_*$ is a learnable weight matrix.
    - $b_*$ is a learnable bias term.
  </details>

  <details>

    <summary>Definition of backpropagation variables</summary>

    - Total loss: $L = \sum_{t=1}^{T} L_t$
    - Gradient of the hidden state: $\delta\mathbf{h}_t = \frac{\partial L}{\partial \mathbf{h}_t}$
    - Gradient of the cell state: $\delta\mathbf{c}_t = \frac{\partial L}{\partial \mathbf{c}_t}$
  </details>

  <details>

    <summary>Backpropagation algorithm (time step t -> t-1)</summary>

    - Initialization

    $$
    \delta\mathbf{h}_{T} = \mathbf{0}, \quad \delta\mathbf{c}_{T} = \mathbf{0}, \quad \mathbf{f}_{T} = \mathbf{0}
    $$

    - **Loop from $t = T - 1$ to $0$**

      1. **Gradient of the current hidden state**

          $$
          \delta\mathbf{h}_t = \frac{\partial L_t}{\partial \mathbf{h}_t} + \delta\mathbf{h}_{\text{next}}
          $$

      2. **Gradient of the current cell state**

          $$
          \delta\mathbf{c}_t = \delta\mathbf{h}_t \odot \mathbf{o}_t \odot (1 - \tanh^2(\mathbf{c}_t)) + \delta\mathbf{c}_{\text{next}} \odot \mathbf{f}_{\text{next}}
          $$

      3. **Calculation of the gating gradient**

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

      5. **Propagation to the previous moment**

          $$
          \delta\mathbf{z}_t = \mathbf{W}_f^\top \delta\mathbf{f}_t + \mathbf{W}_i^\top \delta\mathbf{i}_t + \mathbf{W}_g^\top \delta\mathbf{g}_t + \mathbf{W}_o^\top \delta\mathbf{o}_t
          $$

          $$
          \delta\mathbf{h}_{\text{prev}} = \delta\mathbf{z}_t[1:\dim(\mathbf{h}_{t-1})]
          $$

          $$
          \delta\mathbf{c}_{\text{prev}} = \delta\mathbf{c}_t \odot \mathbf{f}_t
          $$

      6. **Update communication variables**

          $$
          \delta\mathbf{h}_{\text{next}} \leftarrow \delta\mathbf{h}_{\text{prev}}
          $$

          $$
          \delta\mathbf{c}_{\text{next}} \leftarrow \delta\mathbf{c}_{\text{prev}}
          $$

          $$
          \mathbf{f}_{\text{next}} \leftarrow \mathbf{f}_t
          $$

    </details>

  <details>

    <summary>Gradient calculation principle</summary>

    - **Derivation of cell state gradient**

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

    - **Derivation of the forget gate gradient**

      $$
      \delta\mathbf{f}_t = \frac{\partial L}{\partial \mathbf{a}_f^t} = \delta\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t \odot (1 - \mathbf{f}_t)
      $$

    - **Derivation of the parameter gradient**

      $$
      \frac{\partial L}{\partial \mathbf{W}_f} = \sum_{t=1}^{T} \delta\mathbf{f}_t \mathbf{z}_t^\top
      $$

    - **Gradient flow characteristics of LSTM**

      **Long-range dependency handling**

      $$
      \frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_1} = \prod_{k=2}^{T} \mathbf{f}_k \quad \text{ (diagonal matrix)}
      $$

  </details>

  <details>

    <summary>Backpropagation of multi-layer LSTM</summary>
    In a multi-layer LSTM network, the gradient propagation between layers focuses only on the transfer of hidden states (ignoring the internal details of a single layer, such as the gating mechanism or cell state). If:

    - $\mathbf{h}^{(l)}$: hidden state of layer $l$ ($l = 1, 2, \dots, L$, where $L$ is the total number of layers)
    - $L$: loss function
    - $\frac{\partial L}{\partial \mathbf{h}^{(l)}}$: gradient of the loss function with respect to the hidden state of layer $l$

    **Core propagation formula**

    The gradient is propagated from the top layer ($l = L$) to the bottom layer ($l = 1$), and the inter-layer relationship is given by the chain rule:

    $$
    \frac{\partial L}{\partial \mathbf{h}^{(l-1)}} = \frac{\partial L}{\partial \mathbf{h}^{(l)}} \cdot \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}
    $$

    Where:

    - $\frac{\partial L}{\partial \mathbf{h}^{(l)}}$: gradient of the current layer l (obtained from the previous layer through backpropagation)
    - $\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}$: Jacobian matrix of the hidden state of layer l with respect to the hidden state of layer l-1
    - $\cdot$: matrix multiplication (gradient propagation is essentially vector-matrix multiplication)

    That is, the gradient dx of the output of each layer is the gradient dy of the input of the previous layer.
  </details>

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLstmBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnLstmBackward` is called to perform computation.

```Cpp
aclnnStatus aclnnLstmBackwardGetWorkspaceSize(
  const aclTensor     *input,
  const aclTensorList *hx,
  const aclTensorList *params,
  const aclTensor     *dy,
  const aclTensor     *dh,
  const aclTensor     *dc,
  const aclTensorList *i,
  const aclTensorList *j,
  const aclTensorList *f,
  const aclTensorList *o,
  const aclTensorList *h,
  const aclTensorList *c,
  const aclTensorList *tanhc,
  const aclTensor     *batchSizesOptional,
  bool                hasBias,
  int64_t             numLayers,
  double              dropout,
  bool                train,
  bool                bidirectional,
  bool                batchFirst,
  const aclBoolArray  *outputMask,
  aclTensor           *dxOut,
  aclTensor           *dhPrevOut,
  aclTensor           *dcPrevOut,
  aclTensorList       *dparamsOut,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnLstmBackward(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnLstmBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 280px">
  <col style="width: 340px">
  <col style="width: 140px">
  <col style="width: 120px">
  <col style="width: 270px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Fixed-length input sequence of the LSTM, corresponding to x in the formula.</td>
      <td>batch_size indicates the number of sequence groups, time_step indicates the time dimension, and input_size indicates the number of input features.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td><ul>
      <li>If valid batchSizesOptional is passed, the value is [time_step * batch_size, input_size]</li>.
      <li>If the null pointer batchSizesOptional is passed, the value is [time_step, batch_size, input_size] or [batch_size, time_step, input_size]</li></ul></td>.
      <td>√</td>
    </tr>
    <tr>
      <td>hx</td>
      <td>Input</td>
      <td>Initial hidden and cell states of each LSTM layer. It corresponds to h(t-1) and c(t-1) at time 0.</td>
      <td><ul><li>The list contains two elements, h_0 and c_0. </li><li>In the case of multi-layer bidirectional LSTM, each tensor is arranged in the order of bidirectional and then layer-wise along the 0th dimension. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>The shape of each tensor in the list is [D * num_layers, batch_size, hidden_size]</td>.
      <td>√</td>
    </tr>
    <tr>
      <td>params</td>
      <td>Input</td>
      <td>LSTM weight and bias tensor list of each layer, corresponding to w and b in the formula.</td>
      <td><ul><li>If bidirection is True, `D = 2` is used. Otherwise, `D = 1` is used. If hasBiases is True, `B = 2` is used. Otherwise, `B = 1` is used. The list length is specified by D * B * num_layers. </li><li> When both bidirection and hasBias are set to True, the layout is [weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0].</li>
      <li>If the value of hasBias is False, there is no bias item. If the value of bidirection is False, there is no reverse item. </li><li>For multiple layers, the layout is layer-by-layer.</li><li> The data type is the same as that of input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td><ul><li>weight_ih: [4*hidden_size, cur_input_size]</li><li>weight_hh: [4*hidden_size, hidden_size]</li><li>bias_ih: [4*hidden_size]</li><li>bias_hh: [4*hidden_size]</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>Input</td>
      <td>Gradient of the hidden state output by the last layer in the forward direction of LSTM. It corresponds to ∂L/∂h^(l) in the formula.</td>
      <td><ul><li>In bidirectional mode, data is arranged in the last dimension in forward and backward directions. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td><ul>
      <li>If valid batchSizesOptional is passed, the value is [time_step * batch_size, hidden_size * D]</li>.
      <li>If the null pointer batchSizesOptional is passed, the value is [time_step, batch_size, hidden_size * D] or [batch_size, time_step, hidden_size * D]</li></ul></td>.
      <td>√</td>
    </tr>
    <tr>
      <td>dh</td>
      <td>Input</td>
      <td>Gradient of the hidden state output by each layer in the forward direction of LSTM, which is transferred from the next time step at time T. It corresponds to δh_next.</td>
      <td><ul><li>In multi-layer bidirectional mode, data is arranged in the first dimension in the order of bidirectional and then layer-wise. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[numLayers * D, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dc</td>
      <td>Input</td>
      <td>Gradient of the output cell of each LSTM layer at time step T, which is transferred from the next time step. It corresponds to δc_next.</td>
      <td><ul><li>In the case of multi-layer bidirectional LSTM, data is arranged in the order of bidirectional and then layer-wise along dimension 0. </li><li>The data type is the same as that of input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[numLayers * D, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
    <td>i</td>
      <td>Input</td>
      <td>Activation value of the input gate output by each LSTM layer in the forward direction. It corresponds to i in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>g</td>
      <td>Input</td>
      <td>Activation value of the candidate cell state output by each LSTM layer in the forward direction. It corresponds to g in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Input</td>
      <td>Activation value of the forget gate at each layer in the forward LSTM. It corresponds to f in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>o</td>
      <td>Input</td>
      <td>Activation value of the output gate at each layer in the forward LSTM. It corresponds to o in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>h</td>
      <td>Input</td>
      <td>Hidden state of each layer in the forward LSTM. It corresponds to h in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Input</td>
      <td>Final cell state of each layer in the forward LSTM. It corresponds to c in the formula.</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>tanhc</td>
      <td>Input</td>
      <td>Output of the final cell state at each layer in the forward direction of LSTM after being activated by the tanh function. It corresponds to tanh(c).</td>
      <td><ul><li>The length of the list is D x num_layers. </li><li>In the case of multi-layer bidirectional LSTM, tensors are arranged in the order of bidirectional and then multi-layer. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[time_step, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSizesOptional</td>
      <td>Input</td>
      <td>Number of valid sequence batches at each time step of the variable-length LSTM input sequence.</td>
      <td>Supported for variable-length sequences.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[time_step]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias</td>
      <td>Input</td>
      <td>Whether there is a bias b.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numLayers</td>
      <td>Input</td>
      <td>Number of LSTM layers.</td>
      <td>The value is greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>train</td>
      <td>Input</td>
      <td>Indicates whether the scenario is a training scenario.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bidirection</td>
      <td>Input</td>
      <td>Whether it is bidirectional.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchFirst</td>
      <td>Input</td>
      <td>Indicates whether the input data input, y, dy, and dxOut are in the format where the batch dimension is the first dimension.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>Indicates whether to calculate four gradients.</td>
      <td>The array length is 4, which is not supported currently.</td>
      <td>ACL_BOOL_ARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>Output</td>
      <td>Gradient on the input, corresponding to δx in the formula.</td>
      <td><ul><li>The shape is the same as that of the input. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dhPrevOut</td>
      <td>Output</td>
      <td>Gradient of the initial hidden state of each LSTM layer, corresponding to δh_prev when t = 0.</td>
      <td><ul><li>In the case of multi-layer bidirectional LSTM, data is arranged in the order of bidirectional and then layer-wise along the 0th dimension. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[D * num_layers, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dcPrevOut</td>
      <td>Output</td>
      <td>Gradient of the initial cell state of each LSTM layer, corresponding to δc_prev when t = 0.</td>
      <td><ul><li>In the case of multi-layer bidirectional LSTM, data is arranged in the order of bidirectional and then layer-wise along the 0th dimension. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[D * num_layers, batch_size, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dparamsOut</td>
      <td>Output</td>
      <td>Gradient tensor list of the weight and bias. It corresponds to δw and δb in the formula.</td>
      <td><ul><li>The list length is D x B x num_layers. </li><li>The arrangement is the same as that of the input params. </li><li>The data type is the same as that of the input.</li></ul></td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td><ul><li>dweight_ih: [4*hidden_size, cur_input_size]</li><li>dweight_hh: [4*hidden_size, hidden_size]</li><li>dbias: [4*hidden_size]</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the input parameter is aclTensor or aclTensorList and is not batchSizesOptional, the pointer is null.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>If the input parameter is aclTensor or aclTensorList, the data type is not supported.</td>
    </tr>
    <tr>
      <td>If the input parameter type is aclTensor or aclTensorList, the data types are different.</td>
    </tr>
    <tr>
      <td>If the input parameter type is aclTensor or aclTensorList, the shape does not meet the corresponding requirements.</td>
    </tr>
    <tr>
      <td>numLayers is not greater than 0.</td>
    </tr>
  </tbody>
  </table>

## aclnnLstmBackward

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the API aclnnLstmBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>
- **Returns**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnLstmBackward function is implemented in deterministic mode by default.

- Boundary value scenarios:
  - If the input is Inf, the output is NAN.
  - When the input is `NaN`, the output is `NaN`.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm_backward.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize AscendCL.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor, aclFormat format=aclFormat::ACL_FORMAT_ND) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  // Define variables.
  int64_t t = 1;
  int64_t n = 1;
  int64_t inputSize = 8;
  int64_t hiddenSize = 8;

  // Define the shape.
  std::vector<int64_t> xShape = {t, n, inputSize};
  std::vector<int64_t> wiShape = {hiddenSize * 4, inputSize};
  std::vector<int64_t> whShape = {hiddenSize * 4, hiddenSize};
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> yShape = {t, n, hiddenSize};
  std::vector<int64_t> initHShape = {1, n, hiddenSize};
  std::vector<int64_t> initCShape = initHShape; // is the same as initHShape.
  std::vector<int64_t> hShape = yShape;
  std::vector<int64_t> cShape = hShape;
  std::vector<int64_t> dyShape = yShape;
  std::vector<int64_t> dhShape = {1, n, hiddenSize};
  std::vector<int64_t> dcShape = dhShape;
  std::vector<int64_t> iShape = hShape;
  std::vector<int64_t> jShape = hShape;
  std::vector<int64_t> fShape = hShape;
  std::vector<int64_t> oShape = hShape;
  std::vector<int64_t> tanhCtShape = hShape;

  // Shape of the output tensor for backpropagation
  std::vector<int64_t> dwiShape = wiShape; // is the same as wi.
  std::vector<int64_t> dwhShape = wiShape; // is the same as wh.
  std::vector<int64_t> dbShape = bShape; // is the same as b.
  std::vector<int64_t> dxShape = xShape; // is the same as x.
  std::vector<int64_t> dhPrevShape = initHShape; // is the same as initH.
  std::vector<int64_t> dcPrevShape = initCShape; // is the same as initC.

  //Pointer to the device address
  void* xDeviceAddr = nullptr;
  void* wiDeviceAddr = nullptr;
  void* whDeviceAddr = nullptr;
  void* biDeviceAddr = nullptr;
  void* bhDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* hDeviceAddr = nullptr;
  void* cDeviceAddr = nullptr;
  void* dyDeviceAddr = nullptr;
  void* dhDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* iDeviceAddr = nullptr;
  void* jDeviceAddr = nullptr;
  void* fDeviceAddr = nullptr;
  void* oDeviceAddr = nullptr;
  void* tanhCtDeviceAddr = nullptr;

  //Pointer to the output device address for backpropagation
  void* dwiDeviceAddr = nullptr;
  void* dwhDeviceAddr = nullptr;
  void* dbiDeviceAddr = nullptr;
  void* dbhDeviceAddr = nullptr;
  void* dxDeviceAddr = nullptr;
  void* dhPrevDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;

  //Pointer to the ACL tensor.
  aclTensor* x = nullptr;
  aclTensor* wi = nullptr;
  aclTensor* wh = nullptr;
  aclTensor* bi = nullptr;
  aclTensor* bh = nullptr;
  aclTensor* y = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* h = nullptr;
  aclTensor* c = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* dh = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* i = nullptr;
  aclTensor* j = nullptr;
  aclTensor* f = nullptr;
  aclTensor* o = nullptr;
  aclTensor* tanhCt = nullptr;

  // Output the ACL tensor pointer for backpropagation.
  aclTensor* dwi = nullptr;
  aclTensor* dwh = nullptr;
  aclTensor* dbi = nullptr;
  aclTensor* dbh = nullptr;
  aclTensor* dx = nullptr;
  aclTensor* dhPrev = nullptr;
  aclTensor* dcPrev = nullptr;

  std::vector<float> xHostData(xShape[0] * xShape[1] * xShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> wiHostData(wiShape[0] * wiShape[1], 1.0f); // (8+8)*32 = 16*32 = 512 ones
  std::vector<float> whHostData(whShape[0] * whShape[1], 1.0f); // (8+8)*32 = 16*32 = 512 ones
  std::vector<float> biHostData(bShape[0], 1.0f); // 32 ones
  std::vector<float> bhHostData(bShape[0], 1.0f); // 32 ones
  std::vector<float> yHostData(yShape[0] * yShape[1] * yShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> initHHostData(initHShape[0] * initHShape[1] * initHShape[2], 1.0f); // 1*8 = 8 ones
  std::vector<float> initCHostData(initCShape[0] * initCShape[1] * initCShape[2], 1.0f); // 1*8 = 8 ones
  std::vector<float> hHostData(hShape[0] * hShape[1] * hShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> cHostData(cShape[0] * cShape[1] * cShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> dyHostData(dyShape[0] * dyShape[1] * dyShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> dhHostData(dhShape[0] * dhShape[1] * dhShape[2], 1.0f); // 1*8 = 8 ones
  std::vector<float> dcHostData(dcShape[0] * dcShape[1] * dcShape[2], 1.0f); // 1*8 = 8 ones
  std::vector<float> iHostData(iShape[0] * iShape[1] * iShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> jHostData(jShape[0] * jShape[1] * jShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> fHostData(fShape[0] * fShape[1] * fShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> oHostData(oShape[0] * oShape[1] * oShape[2], 1.0f); // 1*1*8 = 8 ones
  std::vector<float> tanhCtHostData;
  tanhCtHostData.reserve(cHostData.size());
  for (const auto& cVal : cHostData) {
      tanhCtHostData.push_back(std::tanh(cVal)); // Apply the tanh function to each c value.
  }
  // Backpropagate the output host data (initialized to 0).
  std::vector<float> dwiHostData(dwiShape[0] * dwiShape[1], 0.0f);
  std::vector<float> dwhHostData(dwhShape[0] * dwhShape[1], 0.0f);
  std::vector<float> dbiHostData(dbShape[0], 0.0f);
  std::vector<float> dbhHostData(dbShape[0], 0.0f);
  std::vector<float> dxHostData(dxShape[0] * dxShape[1] * dxShape[2], 0.0f);
  std::vector<float> dhPrevHostData(dhPrevShape[0] * dhPrevShape[1] * dhPrevShape[2], 0.0f);
  std::vector<float> dcPrevHostData(dcPrevShape[0] * dcPrevShape[1] * dcPrevShape[2], 0.0f);


  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create params aclTensorList.
  ret = CreateAclTensor(wiHostData, wiShape, &wiDeviceAddr, aclDataType::ACL_FLOAT, &wi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(whHostData, whShape, &whDeviceAddr, aclDataType::ACL_FLOAT, &wh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biHostData, bShape, &biDeviceAddr, aclDataType::ACL_FLOAT, &bi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(bhHostData, bShape, &bhDeviceAddr, aclDataType::ACL_FLOAT, &bh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* paramsArray[] = {wi, wh, bi, bh};
  auto paramsList = aclCreateTensorList(paramsArray, 4);

  // Create a y aclTensor.
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create initH aclTensor.
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT, &initH, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create initC aclTensor.
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT, &initC, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* initHcArray[] = {initH, initC};
  auto initHcList = aclCreateTensorList(initHcArray, 2);

  // Create h aclTensor.
  ret = CreateAclTensor(hHostData, hShape, &hDeviceAddr, aclDataType::ACL_FLOAT, &h, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* hArray[] = {h};
  auto hList = aclCreateTensorList(hArray, 1);

  // Create c aclTensor.
  ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT, &c, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* cArray[] = {c};
  auto cList = aclCreateTensorList(cArray, 1);

  // Create dy aclTensor.
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the DH ACL tensor.
  ret = CreateAclTensor(dhHostData, dhShape, &dhDeviceAddr, aclDataType::ACL_FLOAT, &dh, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the DC ACL tensor.
  ret = CreateAclTensor(dcHostData, dcShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the i ACL tensor.
  ret = CreateAclTensor(iHostData, iShape, &iDeviceAddr, aclDataType::ACL_FLOAT, &i, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* iArray[] = {i};
  auto iList = aclCreateTensorList(iArray, 1);

  // Create the j ACL tensor.
  ret = CreateAclTensor(jHostData, jShape, &jDeviceAddr, aclDataType::ACL_FLOAT, &j, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* jArray[] = {j};
  auto jList = aclCreateTensorList(jArray, 1);

  // Create the f ACL tensor.
  ret = CreateAclTensor(fHostData, fShape, &fDeviceAddr, aclDataType::ACL_FLOAT, &f, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* fArray[] = {f};
  auto fList = aclCreateTensorList(fArray, 1);

  // Create an o aclTensor.
  ret = CreateAclTensor(oHostData, oShape, &oDeviceAddr, aclDataType::ACL_FLOAT, &o, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* oArray[] = {o};
  auto oList = aclCreateTensorList(oArray, 1);

  // Create a tanhCt aclTensor.
  ret = CreateAclTensor(tanhCtHostData, tanhCtShape, &tanhCtDeviceAddr, aclDataType::ACL_FLOAT, &tanhCt, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* tanhCtArray[] = {tanhCt};
  auto tanhCtList = aclCreateTensorList(tanhCtArray, 1);

  // Create the backpropagation output tensor.

  // Create a dx aclTensor.
  ret = CreateAclTensor(dxHostData, dxShape, &dxDeviceAddr, aclDataType::ACL_FLOAT, &dx, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a dhPrev aclTensor.
  ret = CreateAclTensor(dhPrevHostData, dhPrevShape, &dhPrevDeviceAddr, aclDataType::ACL_FLOAT, &dhPrev, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the dcPrev aclTensor.
  ret = CreateAclTensor(dcPrevHostData, dcPrevShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev, aclFormat::ACL_FORMAT_NCL);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the dparams aclTensorList.
  ret = CreateAclTensor(dwiHostData, dwiShape, &dwiDeviceAddr, aclDataType::ACL_FLOAT, &dwi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dwhHostData, dwhShape, &dwhDeviceAddr, aclDataType::ACL_FLOAT, &dwh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbiHostData, bShape, &dbiDeviceAddr, aclDataType::ACL_FLOAT, &dbi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(dbhHostData, bShape, &dbhDeviceAddr, aclDataType::ACL_FLOAT, &dbh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  aclTensor* dparamsArray[] = {dwi, dwh, dbi, dbh};
  auto dparamsList = aclCreateTensorList(dparamsArray, 4);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first part of the aclnnLstmBackward API.
  ret = aclnnLstmBackwardGetWorkspaceSize(x, initHcList, paramsList, dy, dh, dc, iList, jList, fList,
    oList, hList, cList ,tanhCtList, nullptr, true, 1, 0, true, false, false, nullptr, dx, dhPrev, dcPrev, dparamsList,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second part of the aclnnLstmBackward API.
  ret = aclnnLstmBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLstmBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  // Print the dparams result.
  auto dwiSize = GetShapeSize(dwiShape);
  std::vector<float> resultDwiData(dwiSize, 0);
  ret = aclrtMemcpy(resultDwiData.data(), resultDwiData.size() * sizeof(resultDwiData[0]), dwiDeviceAddr,
                    dwiSize * sizeof(resultDwiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwiSize; i++) {
    LOG_PRINT("result dwi[%ld] is: %f\n", i, resultDwiData[i]);
  }

  auto dwhSize = GetShapeSize(dwhShape);
  std::vector<float> resultDwhData(dwhSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dwhDeviceAddr,
                    dwhSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dwh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dwhSize; i++) {
    LOG_PRINT("result dwh[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dbiSize = GetShapeSize(bShape);
  std::vector<float> resultDbiData(dbiSize, 0);
  ret = aclrtMemcpy(resultDbiData.data(), resultDbiData.size() * sizeof(resultDbiData[0]), dbiDeviceAddr,
                    dbiSize * sizeof(resultDbiData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbi result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbiSize; i++) {
    LOG_PRINT("result dbi[%ld] is: %f\n", i, resultDbiData[i]);
  }

  auto dbhSize = GetShapeSize(bShape);
  std::vector<float> resultDbhData(dbhSize, 0);
  ret = aclrtMemcpy(resultDbhData.data(), resultDbhData.size() * sizeof(resultDbhData[0]), dbhDeviceAddr,
                    dbhSize * sizeof(resultDbhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dbh result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbhSize; i++) {
    LOG_PRINT("result dbh[%ld] is: %f\n", i, resultDbhData[i]);
  }

  //Print the dx result.
  auto dxSize = GetShapeSize(dxShape);
  std::vector<float> resultDxData(dxSize, 0);
  ret = aclrtMemcpy(resultDxData.data(), resultDxData.size() * sizeof(resultDxData[0]), dxDeviceAddr,
                    dxSize * sizeof(resultDxData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dx result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dxSize; i++) {
    LOG_PRINT("result dx[%ld] is: %f\n", i, resultDxData[i]);
  }

  //Print the dh_prev result.
  auto dhPrevSize = GetShapeSize(dhPrevShape);
  std::vector<float> resultDhPrevData(dhPrevSize, 0);
  ret = aclrtMemcpy(resultDhPrevData.data(), resultDhPrevData.size() * sizeof(resultDhPrevData[0]), dhPrevDeviceAddr,
                    dhPrevSize * sizeof(resultDhPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dh_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dhPrevSize; i++) {
    LOG_PRINT("result dh_prev[%ld] is: %f\n", i, resultDhPrevData[i]);
  }

  //Print the dc_prev result.
  auto dcPrevSize = GetShapeSize(dcPrevShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dc_prev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dc_prev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }

  //Release the aclTensor.
  aclDestroyTensor(x);
  aclDestroyTensor(wi);
  aclDestroyTensor(wh);
  aclDestroyTensor(bi);
  aclDestroyTensor(bh);
  aclDestroyTensor(y);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(h);
  aclDestroyTensor(c);
  aclDestroyTensor(dy);
  aclDestroyTensor(dh);
  aclDestroyTensor(dc);
  aclDestroyTensor(i);
  aclDestroyTensor(j);
  aclDestroyTensor(f);
  aclDestroyTensor(o);
  aclDestroyTensor(tanhCt);
  aclDestroyTensor(dwi);
  aclDestroyTensor(dwh);
  aclDestroyTensor(dbi);
  aclDestroyTensor(dbh);
  aclDestroyTensor(dx);
  aclDestroyTensor(dhPrev);
  aclDestroyTensor(dcPrev);

  // Free the tensor list.
  aclDestroyTensorList(paramsList);
  aclDestroyTensorList(initHcList);
  aclDestroyTensorList(hList);
  aclDestroyTensorList(cList);
  aclDestroyTensorList(iList);
  aclDestroyTensorList(jList);
  aclDestroyTensorList(fList);
  aclDestroyTensorList(oList);
  aclDestroyTensorList(tanhCtList);
  aclDestroyTensorList(dparamsList);

  // Free device resources.
  aclrtFree(xDeviceAddr);
  aclrtFree(wiDeviceAddr);
  aclrtFree(whDeviceAddr);
  aclrtFree(biDeviceAddr);
  aclrtFree(bhDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(hDeviceAddr);
  aclrtFree(cDeviceAddr);
  aclrtFree(dyDeviceAddr);
  aclrtFree(dhDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(iDeviceAddr);
  aclrtFree(jDeviceAddr);
  aclrtFree(fDeviceAddr);
  aclrtFree(oDeviceAddr);
  aclrtFree(tanhCtDeviceAddr);
  aclrtFree(dwiDeviceAddr);
  aclrtFree(dwhDeviceAddr);
  aclrtFree(dbiDeviceAddr);
  aclrtFree(dbhDeviceAddr);
  aclrtFree(dxDeviceAddr);
  aclrtFree(dhPrevDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
