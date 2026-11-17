# chunk_loss

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: This operator fuses the language-model head linear transform with sparse cross-entropy and returns the gradient w.r.t. the hidden state. Given one chunk of hidden states `hidden`, vocabulary weights `weight`, and labels `labels`, it streams over the vocabulary dimension on Device without materializing full logits. Outputs are the scalar `loss`, online states `max_out`/`sum_out`, and the gradient `d_h` w.r.t. `hidden`. The gradient of `weight` is not produced. Callers may split along `(M, H)` and invoke per chunk; with the same `alpha`, summing the chunk `loss` values matches a single full-sequence call.

- Formula:

  `hidden` is $X$ with shape $[M, H]$. `labels` is $y$ with shape $[M]$. `weight` is $W$ with shape $[V, H]$.

  $$
  Z = X W^{\mathsf T}
  $$

  Negative log-likelihood for token $i$:

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

## Function Prototype

```python
cann_ops_nn.chunk_loss(hidden, weight, labels, alpha, ignore_index=-100) -> (Tensor, Tensor, Tensor, Tensor)
```

## Mandatory

>**NOTE:**<br>
>
>- `hidden` has shape `[M, H]`, `weight` has shape `[V, H]`, and `labels` has shape `[M]`.
>- `alpha` scales `loss`/`dZ` in FLOAT32 on Device and must satisfy `alpha > 0`.

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Parameter Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>hidden</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>One chunk of hidden states (X in the formula).</td>
        <td>float16, bfloat16</td>
        <td>[M, H]</td>
    </tr>
    <tr>
        <td>weight</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Vocabulary weights (W). Must match the dtype of hidden.</td>
        <td>float16, bfloat16</td>
        <td>[V, H]</td>
    </tr>
    <tr>
        <td>labels</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Labels (y). Non-ignore_index values must lie in [0, V).</td>
        <td>int64</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>float</td>
        <td>Required</td>
        <td>Scale for loss and dZ; must be greater than 0.</td>
        <td>float</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ignore_index</td>
        <td>int</td>
        <td>Optional</td>
        <td>Ignored label value, default -100. nll and dZ are zero at those positions.</td>
        <td>int</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Return Value

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Output</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Shape</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>loss</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Scalar cross-entropy loss.</td>
        <td>float32</td>
        <td>[]</td>
    </tr>
    <tr>
        <td>max_out</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Online logsumexp max state.</td>
        <td>float32</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>sum_out</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Online logsumexp sum state.</td>
        <td>float32</td>
        <td>[M]</td>
    </tr>
    <tr>
        <td>d_h</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Gradient w.r.t. hidden (dH). Weight gradient is not produced.</td>
        <td>same as hidden</td>
        <td>[M, H]</td>
    </tr>
</tbody>
</table>

## Constraints

- Supported in training scenarios.
- Supports eager mode and TorchAir graph mode.
- `hidden`/`weight` must be 2-D with matching last dim H; `labels` must be 1-D of length M.
- Empty tensors are not supported; LM-head bias and label smoothing are not supported.
- Non-`ignore_index` `labels` values must lie in `[0, V)`; out-of-range behavior is undefined and must be guaranteed by the caller.

## Determinism / Batch Consistency

- Deterministic computation is supported by default.
- Batch consistency is supported by default.

## Call Examples

- Eager mode

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    M, H, V = 128, 256, 1024
    alpha = 1.0 / M
    ignore_index = -100

    hidden = torch.randn(M, H, dtype=torch.float16, device="npu")
    weight = torch.randn(V, H, dtype=torch.float16, device="npu")
    labels = torch.randint(0, V, (M,), dtype=torch.int64, device="npu")

    loss, max_out, sum_out, d_h = cann_ops_nn.chunk_loss(
        hidden, weight, labels, alpha, ignore_index
    )
    print(loss, max_out.shape, sum_out.shape, d_h.shape)
    ```

- Graph mode (TorchAir)

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn

    M, H, V = 128, 256, 1024
    alpha = 1.0 / M

    class Model(torch.nn.Module):
        def forward(self, hidden, weight, labels):
            return cann_ops_nn.chunk_loss(hidden, weight, labels, alpha, -100)

    model = Model().npu()
    compiled = torchair.inference.cache_compile(model)

    hidden = torch.randn(M, H, dtype=torch.float16, device="npu")
    weight = torch.randn(V, H, dtype=torch.float16, device="npu")
    labels = torch.randint(0, V, (M,), dtype=torch.int64, device="npu")
    loss, max_out, sum_out, d_h = compiled(hidden, weight, labels)
    ```
