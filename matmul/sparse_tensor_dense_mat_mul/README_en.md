# SparseTensorDenseMatMul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|<term>Ascend 950PR&950DT products</term>|√|
|<term>Atlas A3 products</term>|×|
|<term>Atlas A2 products</term>|×|
|<term>Atlas 200I/500 A2 inference products</term>|×|
|<term>Atlas inference products</term>|×|
|<term>Atlas training products</term>|×|

## Function

- Operator function: sparse matrix multiplication. It multiplies the sparse matrix x1, represented by x1\_indices, x1\_values, and x1\_shape, by the dense matrix x2, obtaining the result matrix y. The ranks of x1, x2, and y must all be 2.

- Computation formula:

  $$ y = x1 \times x2 $$

  The sparse matrix x1 is described by three parts: the indices of the non-zero elements `x1_indices` (shape [nnz, 2]), the values of the non-zero elements `x1_values` (shape [nnz]), and the matrix shape `x1_shape` (shape [2], with the value [m, n]). The shape of the dense matrix x2 is [n, p].

  When adjoint\_a=false and adjoint\_b=false:

  $$y_{i,j} = \sum_{k} x1_{i,k} \cdot x2_{k,j}$$

  where $x1_{i,k}$ is 0 (not present in x1\_indices) or takes the value at the corresponding position in x1\_values.

  When adjoint\_a=true, x1 is transposed first and then participates in the matrix multiplication; when adjoint\_b=true, x2 is transposed first and then participates in the matrix multiplication.

## Parameters

<table>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1_indices</td>
      <td>Input</td>
      <td>Indices of the non-zero elements in the sparse matrix, with the shape [nnz, 2], where nnz is the number of non-zero elements.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1_values</td>
      <td>Input</td>
      <td>Values of the non-zero elements in the sparse matrix, with the shape [nnz].</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x1_shape</td>
      <td>Input</td>
      <td>Dense shape of the sparse matrix, with the shape [2] and the value [m, n] (if x1 is transposed, that is, adjoint_a is input as true, the value is [n, m]).</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Dense matrix, with the shape [n, p] (if x2 is transposed, the shape is [p, n]).</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output dense matrix, with the shape [m, p], and its data type is consistent with x1_values.</td>
      <td>FLOAT16, FLOAT, INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>adjoint_a</td>
      <td>Attribute</td>
      <td>Whether to transpose the sparse matrix x1. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjoint_b</td>
      <td>Attribute</td>
      <td>Whether to transpose the dense matrix x2. The default value is false.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types of x1\_values and x2 must be consistent.
- x1\_indices supports the INT32 and INT64 types, freely combined with the type of x1\_values.
- The values of m, n, p, nnz, `m*p`, `n*p`, and `nnz*p` must be within the range `[0, INT32_MAX)`.
- The value of nnz must be within the range `[0, m*n]`.
- Deterministic computation is not supported.
- Non-contiguous tensors are not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation  | [test_geir_sparse_tensor_dense_mat_mul](./examples/test_geir_sparse_tensor_dense_mat_mul.cpp) | Calls the SparseTensorDenseMatMul operator through the [operator IR](op_graph/sparse_tensor_dense_mat_mul_proto.h) and other methods. |
