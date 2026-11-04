# SoftmaxFocalLossGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: computes the backward gradient of SoftmaxFocalLoss. The input pred is the probability after softmax, and the output is the gradient of pred.

- Formula:

  Let pred be $p$ (shape $[N, C]$), target be $t$, dout be $\text{d}$, and weight be $w$ (treated as all 1s by default). For the $n$-th sample, first compute four row scalars:

  $$
  \text{WF}_n = \sum_c \alpha (1-p_{n,c})^{\gamma} t_{n,c}, \quad
  \text{WB}_n = \sum_c \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}
  $$

  $$
  \text{CE}_n = \sum_c -\log(p_{n,c}) t_{n,c} w_{n,c}, \quad
  \text{W}_n = \sum_c w_{n,c} t_{n,c}
  $$

  Then combine element-wise:

  $$
  \text{dce}_{n,c} = p_{n,c}\text{W}_n - t_{n,c}w_{n,c}
  $$

  $$
  \text{dwf}_{n,c} = -\gamma \cdot p_{n,c} \cdot \big[(\text{WF}_n - \text{WB}_n) + \alpha (1-p_{n,c})^{\gamma-1} t_{n,c}\big]
  $$

  $$
  \text{grad}_{n,c} = \big(\text{dce}_{n,c}\text{WF}_n + \text{dwf}_{n,c}\text{CE}_n\big) \cdot \text{d}_{n,c} \cdot \text{coef}
  $$

  Here:

  - $N$ is the batch count, and $C$ is the number of classes. The reduction axis is the last dimension.
  - $\alpha$ and $\gamma$ are the attributes alpha and gamma, respectively.
  - When reduction is "mean", $\text{coef} = 1/(N \times C)$. When reduction is "none" or "sum", $\text{coef} = 1$.
  - $(1-p)^{\gamma}$ is evaluated as $\exp(\gamma \log(1-p))$.

## Parameters

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 266px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>pred</td>
      <td>Input</td>
      <td>The probability output by the preceding softmax, the p in the formula. The values fall in the open interval (0, 1). The length of each dimension must be greater than 0 (empty tensors are not supported). The last dimension is the class dimension, and the other dimensions are sample dimensions.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>Input</td>
      <td>The one-hot label, the t in the formula. The shape is consistent with that of pred.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dout</td>
      <td>Input</td>
      <td>The gradient passed in from upstream, the d in the formula. The shape and data type are consistent with those of pred.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Optional input</td>
      <td>The element-wise weight, the w in the formula. The shape and data type are consistent with those of pred. When not passed in, it is treated as all 1s. This behavior is supported only by <term>Ascend 950PR&950DT products</term>, and other products must explicitly pass in weight.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>The weight coefficient of the modulation factor, the alpha in the formula. The default value is 0.25.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Attribute</td>
      <td>The exponent of the modulation factor, the gamma in the formula. The default value is 2.0.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>Attribute</td>
      <td>The reduction method. Supported values: 'none', 'mean', and 'sum' (case-insensitive). The default value is 'mean'. When 'mean' is used, the gradient is multiplied by 1/numel(pred). 'none' and 'sum' do not scale.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Output</td>
      <td>The gradient of pred. The shape and data type are consistent with those of pred.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method | Sample Code | Description |
| -------- | -------- | ---- |
| Graph-mode invocation | [test_geir_softmax_focal_loss_grad.cpp](examples/test_geir_softmax_focal_loss_grad.cpp) | Invokes the SoftmaxFocalLossGrad operator through graph construction using the [operator IR](./op_graph/softmax_focal_loss_grad_proto.h). |
