# SparseSegmentSumGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    ×     |
| <term>Atlas A2 products</term>    |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: SparseSegmentSumGrad is the backward propagation operator of SparseSegmentSum, used to compute the gradient of the sparse segment sum operation.

- Formula:

$$
output[i] = \sum_{j : indices[j] = i} grad[segment\_ids[j]]
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
      <td>grad</td>
      <td>Input</td>
      <td>Output gradient tensor, with shape (N, d1, d2, ...)</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>1-D index tensor, pointing to rows of the original data</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>Input</td>
      <td>1-D segment ID tensor, pointing to rows of grad</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dim0</td>
      <td>Input</td>
      <td>Scalar, the size of the first dimension of the original data</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Gradient of the input data, with shape (output_dim0, d1, d2, ...)</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- grad must be a tensor with at least 1 dimension (rank ≥ 1).
- indices and segment_ids must be 1-D and have the same length.
- Values in indices must be in the range [0, output_dim0).
- Values in segment_ids must be in the range [0, grad.shape[0]).
- output_dim0 must be a non-negative compile-time constant scalar.
- When multiple segments point to the same indices, deterministic gather-add accumulation is used (each output is accumulated sequentially by a single thread, without atomic operations).

## Invocation Description

| Invocation method | Sample code                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE graph mode | [test_geir](./examples/arch35/test_geir_sparse_segment_sum_grad.cpp)   | Builds the graph and invokes the SparseSegmentSumGrad operator through the [operator IR](./op_graph/sparse_segment_sum_grad_proto.h). |
