# Scatter

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function:
  Updates the values in tensor `data` one by one with the values in tensor `updates` according to the specified axis `axis` and indices `indices`.

- Example:
  This operator has three inputs and one attribute: `data`, `updates`, `indices`, and `axis`, where `data` is the tensor to be updated, `updates` is the tensor that stores the update data, `indices` indicates the update positions,
  and `axis` is the specified update dimension. Based on whether `indices` is 1-dimensional or 2-dimensional, there are the following two scenarios (when `indices` is 0-dimensional, it is equivalent to the 1-dimensional scenario with shape (1), in which case axis 0 of `updates` must be 1):

  **Scenario 1:** `indices` is 1-dimensional. The i-th element of `indices` corresponds to the i-th batch of `updates` (axis 0 of `updates`) and indicates the starting offset along the `axis` dimension when this batch is written into the i-th batch of `data`;
  the size s of `updates` along the `axis` dimension is the length written each time, and `indices[i] + s` must not exceed the size of `data` along the `axis` dimension.

  ```text
  Sample input:
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i] + e <= d
  indices:(n,), n <= a
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim = -2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim = -1

  **Scenario 2:** `indices` is 2-dimensional, and axis 1 of the shape is 2. The i-th row of `indices` corresponds to the i-th batch of `updates`: `indices[i][0]` specifies the position along axis 0 of `data` where this batch is written (different i can write into different batches of `data`),
  and `indices[i][1]` specifies the starting offset along the `axis` dimension; the size s of `updates` along the `axis` dimension is the length written each time, and `indices[i][1] + s` must not exceed the size of `data` along the `axis` dimension.

  ```text
  Sample input:
  data:(a, b, c, d)
  if axis = -2: updates:(n, b, e, d), indices[i][1] + e <= c
  if axis = -1: updates:(n, b, c, e), indices[i][1] + e <= d
  indices:(n, 2), indices[i][0] < a
  ```

      data[indices[i][0]][j][indices[i][1]+k][l] = updates[i][j][k][l] # if dim = -2
      data[indices[i][0]][j][k][indices[i][1]+l] = updates[i][j][k][l] # if dim = -1

## Constraints

- Axis 0 of the shape of `updates` is consistent with axis 0 of the shape of `indices`.
- When `indices` is 0-dimensional, axis 0 of the shape of `updates` is 1.
- Axis 0 of the shape of `updates` is less than or equal to axis 0 of the shape of `data`.
- Except for the `axis` dimension and axis 0, all other dimensions of the shapes of `updates` and `data` are the same.
- When the shape of `indices` is two-dimensional, axis 1 of the shape must equal 2.
- When the data type of `indices` is INT32, DtypeSize=4; when it is INT64, DtypeSize=8. IndicesShapeSize is the product of the shape of `indices`, and the ub to be used = IndicesShapeSize * DtypeSize + 224. When the ub is greater than the total ub size available on the corresponding AI processor version, this is not supported.
- When `indices` contains duplicates, the results at the duplicated positions are not guaranteed.
- Deterministic computation: when `indices` contains duplicate values, the results are non-deterministic. If deterministic computation is enabled, the determinism of the results is guaranteed (only the Ascend 950PR&950DT products need to explicitly enable it, and the other supported models are deterministic by default).

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_inplace_scatter_update](examples/test_aclnn_inplace_scatter_update.cpp) | Invokes the Scatter operator through the [aclnnInplaceScatterUpdate](docs/aclnnInplaceScatterUpdate_en.md) interface. |
