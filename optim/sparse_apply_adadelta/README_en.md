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

  Let the k-th index be k in [0, N), where N is the length of `indices`, and the parameter row number be $i = \text{indices}[k]$. In the following formulas, the $_{old}$ subscript denotes the input value before the update, and $\text{grad}[k]$ takes the k-th row of the gradient:

$$
\text{accum}_{new}[i] = \rho \times \text{accum}_{old}[i] + (1 - \rho) \times \text{grad}[k]^2
$$

$$
\text{update} = \frac{\sqrt{\text{accum\_update}_{old}[i] + \epsilon}}{\sqrt{\text{accum}_{new}[i] + \epsilon}} \times \text{grad}[k]
$$

$$
\text{var}_{new}[i] = \text{var}_{old}[i] - \text{update} \times \text{lr}
$$

$$
\text{accum\_update}_{new}[i] = \rho \times \text{accum\_update}_{old}[i] + (1 - \rho) \times \text{update}^2
$$

  Rows not selected by `indices` keep their original values and do not participate in the computation.

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
      <td>Learning rate, a scalar. A 0-D scalar (shape {}) or a 1-D tensor with a single element (shape {1}) is supported.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>Input</td>
      <td>Decay rate, a scalar (same form as lr), with a value range of [0, 1].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Numerical stability parameter, a scalar (same form as lr), usually a very small positive value (for example, 1e-6).</td>
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
      <td>Updated parameter tensor, with the same shape and dtype as the input var, sharing Device memory with the input var (inplace).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>Updated gradient accumulator, sharing Device memory with the input accum (inplace).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>Output</td>
      <td>Updated update-amount accumulator, sharing Device memory with the input accum_update (inplace).</td>
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

- The shapes of var, accum, and accum_update must be identical (equal in every dimension), the rank must be at least 1, and the first dimension is the sparse dimension.
- The first dimension (N) of grad equals the length of indices, and the remaining dimensions are the same as the subsequent dimensions of var.
- indices must be a 1-D tensor, and its dtype supports int32 and int64.
- The values of indices must be within [0, the size of the first dimension of var) and must be unique.
- lr, rho, and epsilon are scalars: a 0-D scalar (shape {}) or a 1-D tensor with a single element (shape {1}). They are float32, the same as var and grad.
- Only the float32 numeric type is supported.
- Dynamic shape (`-1`) and dynamic rank (`-2`) graphs are supported: when the rank of an input is unknown, shape validation is skipped and the output keeps the concrete shape of the corresponding input. The parameter forms in the table above must still be satisfied when the graph is built.
- Empty tensors are supported: when indices is empty (N=0), no row is updated and the operator returns directly; when one dimension of var is 0, it contains no element and no element-wise computation is performed.

## Invocation Description

| Invocation method | Invocation sample | Description |
| :--- | :--- | :--- |
| Graph-mode invocation | [test_geir_sparse_apply_adadelta](./examples/test_geir_sparse_apply_adadelta.cpp) | Invoke the sparse_apply_adadelta operator on the host by building a graph through the GE IR. |
