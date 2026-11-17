# ChunkLoss

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: fuses the language-model head linear transform with sparse cross-entropy, and returns the gradient with respect to the hidden states. Inputs are one chunk of hidden states `hidden`, vocabulary weights `weight`, and labels `labels`. Computation streams along the vocabulary dimension on Device without materializing full logits. Outputs are the scalar `loss`, online states `max`/`sum`, and the gradient `dH` for `hidden`. This operator does not compute the gradient of `weight`. Callers invoke the operator on `(M, H)` chunks; using the same `alpha` across chunks and summing the chunk `loss` values is equivalent to a single full-sequence call. This operator does not take a chunkSize argument.
- Formula:

  `hidden` is $X$ with shape $[M, H]$. `labels` is $y$ with shape $[M]$. `weight` is $W$ with shape $[V, H]$.

  $$
  Z = X W^{\mathsf T}
  $$

  The negative log-likelihood of the $i$-th token is:

  $$
  \mathrm{nll}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignore\_index} \\
  \mathrm{logsumexp}(Z_i) - Z_{i, y_i} & 0 \le y_i < V
  \end{cases}
  $$

  $$
  loss = \alpha \sum_{i=0}^{M-1} \mathrm{nll}_i
  $$

  $$
  \mathrm{dZ}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignore\_index} \\
  \alpha\left(\mathrm{softmax}(Z_i) - e_{y_i}\right) & 0 \le y_i < V
  \end{cases}
  $$

  $$
  \mathrm{dH} = \mathrm{dZ}\, W
  $$

  Where:

  - $M$ is the number of tokens, equal to dimension 0 of `hidden` and the length of `labels`.
  - $H$ is the last dimension of `hidden` and `weight`.
  - $V$ is dimension 0 of `weight`, that is, the vocabulary size.
  - $e_{y_i}$ is a one-hot vector of length $V$ with a 1 only at position $y_i$.
  - $\alpha$ is the attribute `alpha`, used as a FLOAT32 multiplicative normalization factor on Device ($\alpha>0$).
  - $Z$, $\mathrm{logsumexp}$, $\mathrm{softmax}$, and `loss` are accumulated in FLOAT32 on Device.

## Parameters

<table style="table-layout: auto; width: 100%">
  <thead>
    <tr>
      <th style="white-space: nowrap">Parameter name</th>
      <th style="white-space: nowrap">Input/Output/Attribute</th>
      <th style="white-space: nowrap">Description</th>
      <th style="white-space: nowrap">Data type</th>
      <th style="white-space: nowrap">Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>hidden</td>
      <td>Input</td>
      <td>Hidden states of one chunk, corresponding to X in the formula, shape [M, H].</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Language-model head weights, corresponding to W in the formula, shape [V, H]; stored by rows and without bias.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labels</td>
      <td>Input</td>
      <td>Target tokens aligned with hidden, corresponding to y in the formula, shape [M].</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>loss</td>
      <td>Output</td>
      <td>Scalar loss of this chunk, corresponding to loss in the formula; 0-D scalar.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max</td>
      <td>Output</td>
      <td>Per-row maximum of logits, shape [M].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>Output</td>
      <td>Per-row sum of exp(Z - max), shape [M].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dH</td>
      <td>Output</td>
      <td>Gradient of loss with respect to hidden, shape [M, H]; ignored rows have zero gradient.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Attribute</td>
      <td>Multiplicative normalization factor of loss, corresponding to alpha in the formula; must be finite and greater than 0 (typically 1/valid_tokens for mean, 1 for sum).</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignore_index</td>
      <td>Attribute</td>
      <td>Label value to ignore; labels equal to this value do not contribute to loss, and the corresponding dH row is 0.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- Shapes of `hidden`, `weight`, and `labels` must satisfy: `hidden` is $[M, H]$, `weight` is $[V, H]$, and `labels` is $[M]$; $M$, $H$, and $V$ are all greater than or equal to 1.
- `hidden` and `weight` must have the same data type, either FLOAT16 or BFLOAT16.
- Reduction is fixed to summing nll over all tokens and then multiplying by `alpha`. For multi-chunk calls, pass the same `alpha` to each chunk and sum the chunk `loss` values.
- Language-model head bias and label smoothing are not supported; the gradient of `weight` is not produced.
- Non-`ignore_index` values in `labels` must lie in $[0, V)$; out-of-range behavior is undefined and must be guaranteed by the caller.
- The currently delivered implementations are arch22 (Atlas A2 / ascend910b, ascend910_93) and arch35 (Ascend 950 / ascend950).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_chunk_loss.cpp](examples/test_aclnn_chunk_loss_arch22.cpp) | Calls the ChunkLoss operator through the [aclnnChunkLoss](docs/aclnnChunkLoss.md) interface (arch22 sample). |
| aclnn interface  | [arch35/test_aclnn_chunk_loss_arch35.cpp](examples/arch35/test_aclnn_chunk_loss_arch35.cpp) | Calls the ChunkLoss operator through the [aclnnChunkLoss](docs/aclnnChunkLoss.md) interface (arch35 sample). |
| PyTorch interface | [torch_extension](torch_extension) | Calls the ChunkLoss operator through the [chunk_loss](docs/torchapi_chunk_loss_en.md) interface. |
