# ChunkLossGrad

## Applicable Products

| Product | Supported |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Description

- Operator function: Computes the ChunkLoss weight gradient. It recomputes vocabulary logits from the hidden states, projection weights, and saved forward softmax statistics. It outputs `gradWeightOut` after normalization and upstream gradient scaling.
- Formula:

  $$
  \ell_{ij}=\sum_{k=0}^{H-1}h_{ik}W_{jk}
  $$

  $$
  p_{ij}=\frac{\exp(\ell_{ij}-m_i)}{s_i},\qquad
  g_{ij}=\begin{cases}
  0,&y_i=\mathrm{ignoreIndex}\\
  \alpha\left(p_{ij}-\mathbf{1}[j=y_i]\right),&y_i\ne\mathrm{ignoreIndex}
  \end{cases}
  $$

  $$
  D_{jk}=q\sum_{i=0}^{N-1}g_{ij}h_{ik}
  $$

  Here, $N$, $H$, and $V$ denote the token count for this call, hidden dimension, and vocabulary size, respectively. $h$, $W$, and $y$ correspond to `hidden`, `weight`, and `labels`, respectively. $\alpha$ corresponds to `alpha`. $m$ and $s$ correspond to `softmaxMax` and `softmaxSum`, respectively. $q$ corresponds to `gradOutput`, and $D$ corresponds to the output `gradWeightOut`. $\mathbf{1}[j=y_i]$ equals 1 when $j=y_i$ and 0 otherwise.

  `softmaxMax` stores the maximum vocabulary logit for each token from the corresponding forward computation. `softmaxSum` stores the sum of exponentials after subtracting that maximum:

  $$
  m_i=\max_{0\le j<V}\ell_{ij},\qquad
  s_i=\sum_{j=0}^{V-1}\exp(\ell_{ij}-m_i)
  $$

## Parameters

<table style="table-layout: fixed; width: 1200px">
<colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 600px">
  <col style="width: 200px">
  <col style="width: 100px">
</colgroup>
<thead>
  <tr>
    <th>Parameter</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Data Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>gradOutput</td>
    <td>Input</td>
    <td>Upstream loss gradient, corresponding to q in the formula, with shape [1]. The value can be positive, negative, or zero. Pass 1 when differentiating the current loss alone.</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>hidden</td>
    <td>Input</td>
    <td>Hidden states, corresponding to h in the formula, with shape [N,H].</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>weight</td>
    <td>Input</td>
    <td>Projection weights, corresponding to W in the formula, with shape [V,H].</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>labels</td>
    <td>Input</td>
    <td>Target class indices for each token, corresponding to y in the formula, with shape [N].</td>
    <td>INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmaxMax</td>
    <td>Input</td>
    <td>Maximum vocabulary logits per token saved during the forward pass, corresponding to m in the formula, with shape [N].</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>softmaxSum</td>
    <td>Input</td>
    <td>Sums of stabilized exponentials per token saved during the forward pass, corresponding to s in the formula, with shape [N].</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>alpha</td>
    <td>Attribute</td>
    <td>Global normalization coefficient, corresponding to α in the formula.</td>
    <td>FLOAT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ignoreIndex</td>
    <td>Attribute</td>
    <td>Label value to ignore, matching the forward pass, with a default of -100. The entire logit gradient row is zero for matching tokens.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>gradWeightOut</td>
    <td>Output</td>
    <td>Weight gradient, corresponding to D in the formula, with shape [V,H] and the same data type as weight.</td>
    <td>FLOAT16, BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## Constraints

- All tensors are required, use the ND format, and reside on the same device. Empty tensors are not supported. `hidden`, `weight`, and `gradWeightOut` must have the same data type.
- N, H, and V are positive integers no greater than 2147483647. N×H and V×H must not exceed ⌊(2^63−1)/4⌋.
- Labels that are not ignored must satisfy 0 ≤ labels[i] < V.
- `alpha` is the reciprocal of the forward loss normalization denominator. It must be finite, positive, and no greater than the maximum finite FLOAT32 value. It must remain positive after conversion to FLOAT32. Use the same global `alpha` across chunks and accumulate their outputs in the caller.
- `softmaxMax` must be finite, and `softmaxSum` must be finite and positive.
- `gradOutput` has shape [1] and supports positive, negative, and zero values.

## Calling Description

| Calling Method | Sample Code | Description |
| :--- | :--- | :--- |
| aclnn API | [test_aclnn_chunk_loss_backward](examples/test_aclnn_chunk_loss_backward.cpp) | Call the ChunkLossGrad operator using the [aclnnChunkLossBackward](docs/aclnnChunkLossBackward.md) API. |
| PyTorch API | [API implementation](torch_extension/chunk_loss_grad.py) | Call `cann_ops_nn.chunk_loss_grad` or `torch.ops.cann_ops_nn.chunk_loss_grad`. |
