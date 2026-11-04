# SparseSegmentMeanGrad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

- Operator function: Computes the backward gradient of SparseSegmentMean. In the forward pass, SparseSegmentMean averages the rows selected by `indices` according to `segment_ids`. In the backward pass, the gradient of each segment needs to be averaged by the number of times the segment is referenced and then scattered back to the output rows specified by `indices`.
- Formula:

  Let the length of the first dimension of `x` be $M$ (number of segments), the lengths of `indices` and `segment_ids` be $N$, and the length of the first dimension of the output `y` be `output_dim0`. First, count the number of times each segment is referenced:

  $$
  c_s = \sum_{j=0}^{N-1} [\, \mathrm{segment\_ids}_j = s \,], \quad s = 0, 1, \ldots, M-1
  $$

  Then compute the average by the reference count and scatter it back to the output:

  $$
  y_{i} = \sum_{j=0}^{N-1} [\, \mathrm{indices}_j = i \,] \cdot \frac{x_{\mathrm{segment\_ids}_j}}{\max(c_{\mathrm{segment\_ids}_j},\ 1)}, \quad i = 0, 1, \ldots, \mathrm{output\_dim0}-1
  $$

  Here, $x_s$ and $y_i$ denote the $s$-th row of `x` and the $i$-th row of `y` respectively. $[\,\cdot\,]$ takes 1 when the condition holds and 0 otherwise. Output rows not referenced by any `indices` have the value 0.

## Parameters

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 400px">
  <col style="width: 250px">
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
      <td>x</td>
      <td>Input</td>
      <td>Gradient propagated back from forward SparseSegmentMean, corresponding to `x` in the computation formula. It has at least 1 dimension, and the length of the first dimension is the number of segments <i>M</i>.</td>
      <td>FLOAT, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Indexes used by forward SparseSegmentMean, corresponding to `indices` in the computation formula. They specify the row of `y` to which the gradient of each segment is scattered. 1-dimensional tensor, with the value range [0, `output_dim0`).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>segment_ids</td>
      <td>Input</td>
      <td>Segment numbers used by forward SparseSegmentMean, corresponding to `segment_ids` in the computation formula. 1-dimensional tensor, with the same number of elements as `indices` and the value range [0, <i>M</i>).</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_dim0</td>
      <td>Input</td>
      <td>Length of the first dimension of the output `y`, corresponding to `output_dim0` in the computation formula. Scalar tensor.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Scattered gradient, corresponding to `y` in the computation formula. It has at least 1 dimension, and the shape is `output_dim0` concatenated with the dimensions of `x` except the first dimension.</td>
      <td>FLOAT, FLOAT16, DOUBLE</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- The data types of `x` and `y` must be the same.
- In the operator IR, `x` and `y` are declared to support BFLOAT16, but the current AI CPU implementation does not cover this data type.

## Constraints

- None of the four inputs supports an empty tensor.
- When the value of `indices` exceeds [0, `output_dim0`) or the value of `segment_ids` exceeds [0, <i>M</i>), the operator returns a parameter check failure.
- Output rows not referenced by any `indices` are set to 0 instead of keeping the original values.

## Invocation Description

| Invocation method   | Invocation sample                                                                                | Description                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Graph-mode invocation | [test_geir_sparse_segment_mean_grad](./examples/test_geir_sparse_segment_mean_grad.cpp) | Calls the SparseSegmentMeanGrad operator by building a graph through the [operator IR](./op_graph/sparse_segment_mean_grad_proto.h). |
