# SparseFillEmptyRows

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: Checks the 0th dimension of a sparse tensor row by row. When a row does not contain any non-zero element, the operator adds an element at the position `[row, 0, ...]` with the value `default_value` for that row in the output sparse tensor.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `indices` | Input | Two-dimensional tensor with the shape `[N, rank]`. Each row represents the coordinates of a sparse element in the dense tensor. | INT64 | ND |
| `values` | Input | One-dimensional tensor with the shape `[N]`, representing the value corresponding to each sparse element in `indices`. | BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8 | ND |
| `dense_shape` | Input | One-dimensional tensor with the shape `[rank]`, representing the shape of the dense tensor corresponding to the sparse tensor. `dense_shape[0]` denotes the number of rows. | INT64 | ND |
| `default_value` | Input | Scalar tensor, representing the default value written to `y_values` when empty rows are filled. | BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8 | ND |
| `y_indices` | Output | Sparse indexes after empty rows are filled, with the shape `[N + empty_row_count, rank]`. | INT64 | ND |
| `y_values` | Output | Sparse values after empty rows are filled, with the shape `[N + empty_row_count]`. The data type is consistent with `values`. | BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8 | ND |
| `empty_row_indicator` | Output | One-dimensional Boolean tensor with the shape `[dense_shape[0]]`. `true` indicates that the corresponding row is empty in the input sparse tensor. | BOOL | ND |
| `reverse_index_map` | Output | One-dimensional tensor with the shape `[N]`. `reverse_index_map[i]` indicates the position of the `i`-th input element in the output. | INT64 | ND |

## Constraints

- `indices` must be a two-dimensional tensor, and the size of the second dimension is `rank`.
- The number of elements in `values` must be consistent with `N`, the first dimension of `indices`.
- `dense_shape` must be a one-dimensional tensor, and the number of elements is `rank`.
- `indices[i, 0]` must satisfy `0 <= indices[i, 0] < dense_shape[0]`.
- When `dense_shape[0] == 0`, `N` must be 0.
- The data types of `values`, `default_value`, and `y_values` need to be consistent.
- This operator is implemented on the AI CPU.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_sparse_fill_empty_rows](./examples/test_geir_sparse_fill_empty_rows.cpp) | Calls the SparseFillEmptyRows operator by building a graph through the [operator IR](./op_graph/sparse_fill_empty_rows_proto.h). |
