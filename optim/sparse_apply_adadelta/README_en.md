# SparseApplyAdadelta

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

- Operator function: SparseApplyAdadelta is the sparse version of the AdaDelta optimization algorithm, and is used to update sparse parameters such as Embedding with adaptive learning rates during training. It updates only the rows specified by `indices`, and is applicable to gradient update scenarios of large-scale sparse parameters such as the Embedding layer.

- Formula:

$$
\text{accum}[i] = \rho \times \text{accum}[i] + (1 - \rho) \times \text{grad}[i]^2
$$

$$
\text{update} = \sqrt{\frac{\text{accum\_update}[i] + \epsilon}{\text{accum}[i] + \epsilon}} \times \text{grad}[i]
$$

$$
\text{var}[i] = \text{var}[i] - \text{update} \times \text{lr}
$$

$$
\text{accum\_update}[i] = \rho \times \text{accum\_update}[i] + (1 - \rho) \times \text{update}^2
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
      <td>Parameter tensor to be updated, whose first dimension is the sparse dimension.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>Exponential moving average of the squared gradient, with the same shape and dtype as var.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>Input</td>
      <td>Exponential moving average of the squared update amount, with the same shape and dtype as var.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate, a scalar.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>Input</td>
      <td>Decay rate, a scalar, with a value range of [0, 1].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Numerical stability parameter, a scalar, usually very small (for example, 1e-6).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient tensor, whose first dimension N is the length of indices.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index vector, which specifies the rows to be updated. Value range: [0, the size of the first dimension of var). The values must be unique, and out-of-bounds indices are not supported.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated parameter tensor, with the same shape and dtype as the input var.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>Updated gradient accumulator.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>Output</td>
      <td>Updated update-amount accumulator.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Specifies whether to use a lock to protect variable updates (the lock mechanism is not implemented on the NPU). Default: false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The shapes of var, accum, and accum_update must be the same.
- The first dimension (N) of grad equals the length of indices, and the remaining dimensions are the same as the subsequent dimensions of var.
- The values of indices must be unique, and out-of-bounds indices are not supported.
- Only the float32 numeric type is supported, and indices supports int32 and int64.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_sparse_apply_adadelta](./examples/test_geir_sparse_apply_adadelta.cpp) | Invoke the sparse_apply_adadelta operator on the host by building a graph through the GE IR. |
