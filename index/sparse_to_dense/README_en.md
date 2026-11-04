# SparseToDense

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    ×    |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: Converts a sparse representation into a dense tensor.

- Example:

  ```cpp
  # If sparse_values is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_values is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_values is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ... , sparse_indices[i][d-1]] = sparse_values[i]
  ```

  The following requirements must be met during computation:
  - `sparse_indices` can only be 0/1/2-dimensional.
  - `sparse_values` can only be 0/1-dimensional.
  - When `sparse_values` is 1-dimensional, the `size` of `sparse_values` must equal the length of axis 0 of `sparse_indices`.
  - The `size` of `default_value` must be 1.
  - The indexes corresponding to each row of `sparse_indices` are unique and sorted in ascending order.
  - Each row of indexes in `sparse_indices` must be valid indexes.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>indices</td>
      <td>Input</td>
      <td>`sparse_indices` in the formula, which is an aclTensor on the device.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_shape</td>
      <td>Input</td>
      <td>Specifies the shape and size of the output tensor. The data type is the same as that of indices.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>values</td>
      <td>Input</td>
      <td>`sparse_values` in the formula, which is an aclTensor on the device.</td>
      <td>UINT8, INT8, UINT16, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, BFLOAT16.</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>default_value</td>
      <td>Input</td>
      <td>`default_value` in the formula, which is an aclTensor on the device. The data type is the same as that of values.</td>
      <td>UINT8, INT8, UINT16, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>validate_indices</td>
      <td>Attribute</td>
      <td>A Boolean value on the host, which determines whether to validate the validity of indices.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>The output in the formula. The data type is the same as that of values.</td>
      <td>UINT8, INT8, UINT16, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- `indices` must be a 0/1/2-dimensional tensor.
- `values` must be a 0/1-dimensional tensor. When it is 1-dimensional, the number of elements must equal the length of the first dimension of `indices`.
- `default_value` must be a scalar and have the same data type as `values`.
- When `validate_indices` is true, the indexes corresponding to each row of `indices` must not repeat and must be sorted in ascending order.
- When `validate_indices` is true, the indexes in each row of `indices` must be within the valid range.
- In the AI Core implementation of Ascend 950PR/950DT, `validate_indices` currently does not take effect. When you use this path, the caller must ensure that `indices` meets the preceding order, uniqueness, and valid range requirements.
- The AI CPU implementation does not support BFLOAT16 for `values`, `default_value`, or `y`.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_sparse_to_dense](./examples/test_geir_sparse_to_dense.cpp) | Calls the SparseToDense operator through graph construction with the [operator IR](./op_graph/sparse_to_dense_proto.h). |
