# SparseSegmentMean

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

- Operator function: Selects several rows from `x` according to `indices` and the ordered `segment_ids`, and averages the rows within the same segment along dimension 0.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ---- | ---- | ---- | ---- | ---- |
| `x` | Input | The input tensor, with a rank of at least 1. | FLOAT, DOUBLE, FLOAT16 | ND |
| `indices` | Input | A one-dimensional tensor, representing the row indices selected from dimension 0 of `x`. | INT32, INT64 | ND |
| `segment_ids` | Input | A one-dimensional tensor whose number of elements is consistent with `indices`; the values must be non-negative and arranged in non-decreasing order. | INT32, INT64 | ND |
| `y` | Output | The output tensor, whose data type is consistent with `x`, whose dimension 0 size is the last element of `segment_ids` plus 1, and whose remaining dimensions are consistent with `x`. | FLOAT, DOUBLE, FLOAT16 | ND |

## Constraints

- `x`, `indices`, and `segment_ids` cannot be empty tensors.
- The rank of `x` must be greater than or equal to 1.
- The numbers of elements of `indices` and `segment_ids` must be the same.
- The values in `indices` must be less than the size of dimension 0 of `x`.
- `segment_ids` must be non-negative and arranged in non-decreasing order.
- The data types of `x` and `y` must be consistent.
- This operator supports the AI CPU implementation, and the AI CPU implementation supports FLOAT, DOUBLE, and FLOAT16.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---- | ---- | ---- |
| Graph-mode invocation | [test_geir_sparse_segment_mean](./examples/test_geir_sparse_segment_mean.cpp) | Calls the SparseSegmentMean operator by building a graph through the [operator IR](./op_graph/sparse_segment_mean_proto.h). |
