# GemmV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: computes α multiplied by the product of A and B, then sums the result with the product of β and the input C.
- Formula:

  $$
  out=α(A @ B) + βC
  $$

  Here, $op(A)$, $op(B)$, and $op(C)$ are matrices with dimensions $(M, K)$, $(K, N)$, and $(M, N)$, respectively. $α$ and $β$ are one-dimensional vectors.

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
      <td>a</td>
      <td>Input</td>
      <td>The left matrix in the matrix multiplication.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>The right matrix in the matrix multiplication.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>The one-dimensional vector multiplied by the matrix multiplication result of a and b.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>The one-dimensional vector multiplied by input c.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Output</td>
      <td>Both input and output, performing in-place accumulation.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transposition scenario.

## Invocation Description

| Invocation method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation  | [test_geir_gemm_v2](./examples/test_geir_gemm_v2.cpp) | Invokes the GemmV2 operator through the [operator IR](op_graph/gemm_v2_proto.h) and other methods. |
