# SparseApplyAdagradV2

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Description: SparseApplyAdagradV2 is the sparse version of the AdagradV2 optimizer operator. It updates the parameter variable and accumulator in place based on the sparse gradient and index vector.

- Formula:

$$
\text{accum}[\text{index}] \mathrel{+}= \text{grad}[i] \times \text{grad}[i] \quad (\text{when update\_slots=True})
$$

$$
\text{var}[\text{index}] \mathrel{-}= \text{lr} \times \frac{\text{grad}[i]}{\sqrt{\text{accum}[\text{index}]} + \epsilon}
$$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>Parameter variable, with the shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>Gradient square accumulator, with the shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate scalar</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Constant factor scalar, used to avoid division by zero</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Sparse gradient, with the shape (M, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Sparse index vector, with the shape (M)</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated parameter variable, with the shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>Updated accumulator, with the shape (N, C)</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Whether to lock during the update (usually ignored in NPU scenarios)</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>Attribute</td>
      <td>Whether to update the accumulator (when True, accum+=grad^2)</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of var and accum must be the same
- lr and epsilon must be scalars (with an empty shape or [1])
- indices must be a 1D vector
- The first dimension of grad must be the same as the first dimension of indices, and the remaining dimensions must be the same as the corresponding dimensions of var
- The dtypes of var, accum, lr, epsilon, and grad must be the same
- The dtype of indices is int32
- Out-of-bounds indices (< 0 or >= the number of rows of var) are skipped without reporting an error
- When indices contains duplicate index values, the result order is not preserved

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="./examples/test_geir_sparse_apply_adagrad_v2.cpp">test_geir_sparse_apply_adagrad_v2</a></td>
    <td>Refer to <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
