# SparseApplyFtrlV2

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

- **Operator function**: performs the Ftrl-proximal V2 optimization algorithm update on sparse gradients at specified indices, including the `l2_shrinkage` regularization term. It targets large-scale sparse feature training scenarios.

- **Formula**:

For each index `i` in `indices`:

$$
\begin{aligned}
\text{grad\_shrink} &= \begin{cases} \text{grad}[i] & l2\_shrinkage = 0 \\ \text{grad}[i] + 2 \times l2\_shrinkage \times \text{var}[i] & \text{otherwise} \end{cases} \\
\text{accum}_{\text{new}}[i] &= \text{accum}[i] + \text{grad}[i]^2 \\
\text{accum\_pow}[i] &= \text{accum}[i]^{-lr\_power} \\
\text{accum\_new\_pow}[i] &= \text{accum}_{\text{new}}[i]^{-lr\_power} \\
\text{delta} &= (\text{accum\_pow}[i] - \text{accum\_new\_pow}[i]) / lr \\
\text{linear}_{\text{new}}[i] &= \text{linear}[i] + \text{grad\_shrink} + \text{delta} \times \text{var}[i] \\
\text{l1\_adjust} &= \text{clamp}(\text{linear}_{\text{new}}[i], -l1, l1) \\
x &= \text{l1\_adjust} - \text{linear}_{\text{new}}[i] \\
y &= \text{accum\_new\_pow}[i] / lr + 2 \times l2 \\
\text{var}_{\text{new}}[i] &= x / y
\end{aligned}
$$

> Usually `lr_power = -0.5`. Only the rows specified in `indices` are updated, and the other rows stay unchanged. If `indices[i]` is out of bounds (<0 or >=N), that row is skipped.

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
      <td>The parameter to be optimized, shape [N, D1, D2, ...]</td>
      <td>float32, float16, bfloat16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>The accumulation of squared gradients, with the same shape as var</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>Input</td>
      <td>The accumulation of linear terms, with the same shape as var</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>The sparse gradient, shape [M, D1, D2, ...]</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>The sparse indices, a 1D vector, shape [M]</td>
      <td>int32, int64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>The learning rate, a scalar</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>Input</td>
      <td>The L1 regularization coefficient, a scalar</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>Input</td>
      <td>The L2 regularization coefficient, a scalar</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2_shrinkage</td>
      <td>Input</td>
      <td>The L2 shrinkage coefficient, a scalar</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr_power</td>
      <td>Input</td>
      <td>The learning rate power, a scalar</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>The updated parameter to be optimized, with the same shape/dtype as input var</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>The updated accumulation of squared gradients, with the same shape/dtype as input accum</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>Output</td>
      <td>The updated accumulation of linear terms, with the same shape/dtype as input linear</td>
      <td>Same as var</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Whether to use a lock mechanism (multi-thread safe). Default value: false. The current implementation does not read this attribute; it is kept only for interface compatibility</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of var, accum, and linear must be exactly the same.
- var is at least 1D (N >= 1). In the ND format it has at most 8 dimensions (1 <= rank <= 8).
- indices must be a 1D vector.
- grad.dim(0) must equal indices.dim(0) (that is, M = len(indices)).
- var.shape[1:] must equal grad.shape[1:], that is, the trailing dimensions match.
- lr, l1, l2, l2_shrinkage, and lr_power must be scalars (with shape [] or [1]).
- The values in indices must be within the range [0, var.shape[0]) and must be unique. Because indices must be unique, M = len(indices) must not exceed var.shape[0] (M <= N). If indices contains duplicate values, the results are unpredictable (consistent with the TensorFlow behavior).
- The dtypes of all data inputs (var/accum/linear/grad/lr/l1/l2/l2_shrinkage/lr_power) must be the same.
- When indices is empty (len(indices) == 0), the computation is skipped, and var/accum/linear stay unchanged.
- lr should be greater than 0; otherwise the denominator is 0 and Inf is produced.
- lr_power should not be greater than 0 (usually -0.5). accum should be non-negative (the implementation clamps negative accum to 0 before the power operation, to avoid NaN).
- The values of all inputs (including scalar hyperparameters) should be finite and should not contain NaN/Inf or extreme values beyond the representation range of the dtype. Otherwise the intermediate quantities (grad², accum^(-lr_power)) overflow, and the results are not guaranteed to match the reference.
- To obtain finite and numerically stable results, the recommended value ranges are: `lr∈(0,1]`, `l1/l2/l2_shrinkage∈[0,1]`, `lr_power∈[-1,0)`, and `accum∈[1e-3,100]`. For `var/linear/grad`, the recommended magnitude is no more than 10 for fp16/bf16 and no more than 1e3 for fp32 (the combination with `accum` and `lr_power` must still not overflow).
- fp16/bf16 is internally computed as fp32 and then cast back to the original type.

## Invocation Description

| Invocation method | Sample code | Description |
|---|---|---|
| GE graph mode | [test_geir_sparse_apply_ftrl_v2](./examples/arch35/test_geir_sparse_apply_ftrl_v2.cpp) | Calls the SparseApplyFtrlV2 operator by building the graph through the [operator IR](./op_graph/sparse_apply_ftrl_v2_proto.h). |
