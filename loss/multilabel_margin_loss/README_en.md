# MultilabelMarginLoss

English | [简体中文](./README.md)


## Applicable Products

| Product | Support |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the multilabel margin loss between the input `x` (prediction scores) and `target` (multilabel indices), and outputs the target bitmask `is_target`.
- Formula:

  For each sample `n` (with the number of classes `C`), let the valid label set of the sample be `T` (the elementwise values of `target`, terminated by the first `-1`; the valid values are the label indices). When `reduction` is `'none'`:

  $$
  \text{loss}[n]=\frac{1}{C}\sum_{j\in T}\sum_{i\notin T}\max\bigl(0,\ 1-(x[n,j]-x[n,i])\bigr)
  $$

  When `reduction` is `'mean'` or `'sum'`, a mean or sum reduction is further applied to the per-sample losses.

  The operator also outputs `is_target`: `is_target[n,i] = 1` (when `i∈T`), otherwise `0`. It is a 0/1 mask derived from `target`.

  Here:
  - `x`: input prediction tensor, with shape `[N,C]` (2D) or `[C]` (1D).
  - `target`: ground-truth multilabel indices, with the same shape as `x`, where each row is padded with `-1` after the valid labels.
  - `is_target`: target bitmask, with the same shape as `target`.

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
    <td>x</td>
    <td>Input</td>
    <td>`x` in the formula, the prediction score tensor. The shape is `[N,C]` (2D) or `[C]` (1D). An empty tensor supports only an empty `N` axis (`[0,C]`, requiring `C≥1`, empty in and empty out); `C=0` is not supported (it makes `1/C` meaningless and reports `161002`).</td>
    <td><term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: BFLOAT16, FLOAT16, FLOAT32<br><term>Atlas training products</term>: FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>target</td>
    <td>Input</td>
    <td>The multilabel indices in the formula, with the same shape as `x`. The element values are in `[-1, C-1]`, and `-1` is the padding/terminator.</td>
    <td>INT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Loss output; `reduction=none` with 2D gives `[N]`, otherwise a scalar `[1]`.</td>
    <td>Consistent with `x`. <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: BFLOAT16, FLOAT16, FLOAT32<br><term>Atlas training products</term>: FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>is_target</td>
    <td>Output</td>
    <td>Target 0/1 bitmask, with the same shape as `target`; the mask values are derived from `target` (0/1). The dtype supports INT32 or consistency with `x`.</td>
    <td>INT32, or consistent with `x`. For the latter, the values are graded by product: <term>Ascend 950PR&950DT products</term>, <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: BFLOAT16, FLOAT16, FLOAT32<br><term>Atlas training products</term>: FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduction</td>
    <td>Attribute</td>
    <td>Reduction method, supporting `'none'`, `'mean'`, and `'sum'`. The default value is `'mean'`.</td>
    <td>STRING</td>
    <td>-</td>
    </tr>

</tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_multilabel_margin_loss](examples/test_aclnn_multilabel_margin_loss.cpp) | Calls the MultilabelMarginLoss operator through the [aclnnMultilabelMarginLoss](docs/aclnnMultilabelMarginLoss_en.md) interface. |
| Graph mode (GE)  | [test_geir_multilabel_margin_loss](examples/test_geir_multilabel_margin_loss.cpp) | Calls the MultilabelMarginLoss operator by building a graph through [operator IR](op_graph/multilabel_margin_loss_proto.h) (the `is_target` output is INT32). |
