# QuantMatmulReduceSum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|×|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: Performs the quantized grouped matrix computation, then adds the matrix computation results of all groups and outputs them
- Formula:

  $$
  out = \sum_{i=0}^{batch}(x1_i @ x2_i) * x1Scale * x2Scale
  $$

  Here, $x1$, $x2$, and $out$ are matrices with dimensions $(batch, M, K)$, $(batch, K, N)$, and $(M, N)$ respectively. $x1Scale$ is a two-dimensional scaling tensor with the shape $(batch, M)$, providing a scaling value for each row of each batch; $x2Scale$ is a one-dimensional scaling vector with the shape $(N,)$.

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
      <td>x1</td>
      <td>Input</td>
      <td>The left matrix in the matrix multiplication.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>The right matrix in the matrix multiplication.</td>
      <td>INT8</td>
      <td>FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>x1Scale</td>
      <td>Input</td>
      <td>The two-dimensional scaling tensor with the shape (batch, M), which provides a scaling value for each row of each batch.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2Scale</td>
      <td>Input</td>
      <td>The one-dimensional vector that column-scales the matrix multiplication result.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>The result matrix.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Empty tensors are not supported.
- The left and right matrices do not support non-contiguous tensors.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn invocation  | [test_aclnn_quant_matmul_reduce_sum](examples/test_aclnn_quant_matmul_reduce_sum_weight_nz.cpp) | Calls the QuantMatmulReduceSum operator through the [aclnnQuantMatmulReduceSumWeightNz](docs/aclnnQuantMatmulReduceSumWeightNz_en.md) interface and other methods. |
