# GemmV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 910_95 AI processor|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: Computes α multiplied by the product of A and B, and then adds the product of β and the input C.
- Formula:

  $$
  out=α(A @ B) + βC
  $$
  Here, $op(A)$, $op(B)$, and $op(C)$ are matrices with dimensions $(M, K)$, $(K, N)$, and $(M, N)$ respectively. $α$ and $β$ are scalar constants.

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
      <td>Left matrix in the matrix multiplication operation.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Right matrix in the matrix multiplication operation.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>Input</td>
      <td>Scalar constant multiplied with the matrix multiplication result of a and b.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>Scalar constant multiplied with the input c.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Output</td>
      <td>Input and output; accumulation is performed in place.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported; [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 170px">
    <col style="width: 300px">
    <col style="width: 430px">
    </colgroup>
    <thread>
      <tr>
        <th>Invocation method</th>
        <th>Sample code</th>
        <th>Description</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>aclnn invocation</td>
        <td>test_aclnn_gemmv3</td>
        <td>Invokes the Gemmv3 operator through the aclnnGemmv3 interface with reference to the sample operator</td>
      </tr>
  </tbody></table>
