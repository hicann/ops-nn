# SparseApplyProximalAdagrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- **Operator function**: Performs the ProximalAdagrad optimization algorithm update on sparse parameters. The algorithm combines Adagrad (adaptive learning rate) and the FOBOS proximal update (L1 soft threshold + L2 weight decay), and is dedicated to the training update of sparse parameters (such as large-scale Embedding tables). Only the rows specified by `indices` are updated, and the remaining rows stay unchanged.

- **Formula**:

For each index `i` in `indices`:

$$
\begin{aligned}
\text{accum}_{\text{new}}[i] &= \text{accum}_{\text{old}}[i] + \text{grad}[i]^2 \\
\text{lr}_1 &= \frac{\text{lr}}{\sqrt{\text{accum}_{\text{new}}[i]}} \\
\text{prox\_var} &= \text{var}_{\text{old}}[i] - \text{grad}[i] \times \text{lr}_1 \\
\text{lr}_2 &= \frac{1}{1 + \text{l}_2 \times \text{lr}_1}
\end{aligned}
$$

If $\text{l}_1 > 0$:
$$
\text{var}_{\text{new}}[i] = \text{sign}(\text{prox\_var}) \times \max(|\text{prox\_var}| - \text{lr}_1 \times \text{l}_1, 0) \times \text{lr}_2
$$

Otherwise ($\text{l}_1 = 0$):
$$
\text{var}_{\text{new}}[i] = \text{prox\_var} \times \text{lr}_2
$$

> ⚠️ The denominator does not contain epsilon. The initial value of accum must be non-zero to avoid division by zero.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
      <td>var</td>
      <td>Input</td>
      <td>Tensor of parameters to be optimized.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>Accumulated gradient square tensor, with the same shape as var.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate scalar.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>Input</td>
      <td>L1 regularization coefficient scalar (≥0).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>Input</td>
      <td>L2 regularization coefficient scalar (≥0).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Sparse gradient tensor. shape[0] equals len(indices), and the trailing dimensions are the same as var.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Sparse index vector, specifying the rows to update in var/accum (values must be unique).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated parameter tensor (inplace semantics, with the same shape/dtype as input var).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>Updated accumulated gradient square tensor (inplace semantics, with the same shape/dtype as input accum).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>If True, the updates of var/accum are protected by a lock; if False, the behavior is undefined but lock contention may be reduced. Default False.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- var and accum must have the same shape.
- var has at least 1 dimension, and indices must be a 1D vector.
- grad.shape[0] == len(indices), and var.shape[1:] == grad.shape[1:].
- lr/l1/l2 are scalars (rank==0 or shape==[1]).
- Values in indices should be in the range [0, var.shape[0]) and should be unique. If indices contains duplicate values, the result is unpredictable (consistent with TensorFlow behavior).
- The dtypes of all data inputs (var/accum/lr/l1/l2/grad) must be exactly the same.
- fp16/bf16 are computed internally in fp32 and cast back to the original type.
- The initial value of accum must be greater than 0 (the denominator does not contain epsilon).

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_sparse_apply_proximal_adagrad](./examples/arch35/test_geir_sparse_apply_proximal_adagrad.cpp) | Calls the SparseApplyProximalAdagrad operator by building a graph with the [operator IR](./op_graph/sparse_apply_proximal_adagrad_proto.h). |
