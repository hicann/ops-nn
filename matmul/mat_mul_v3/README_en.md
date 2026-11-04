# MatMulV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|×|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: Performs general matrix multiplication.
- Formula:

  $$
  C=op(A) @ op(B) + bias
  $$
  Here, $op(A)$, $op(B)$, and $op(C)$ are matrices with dimensions $(M, K)$, $(K, N)$, and $(M, N)$, respectively. $bias$ is a vector with dimension $(1, N)$ or $(N)$.

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
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>b</td>
      <td>Input</td>
      <td>Right matrix in the matrix multiplication operation.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Bias accumulated after the matrix multiplication operation, corresponding to bias in the formula.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>c</td>
      <td>Output</td>
      <td>Computation result of the general matrix multiplication operation.</td>
      <td>FLOAT16, BFLOAT16, FLOAT32</td>
      <td>ND, FRACTAL_NZ</td>
    </tr>
  </tbody></table>

- Atlas inference products: Only the FLOAT16 data format is supported, and input b supports only the FRACTAL_NZ format.
- Atlas A2 products: The FRACTAL_NZ format is not supported, and bias does not support the BFLOAT16 data format.
- Atlas A3 products: The FRACTAL_NZ format is not supported, and bias does not support the BFLOAT16 data format.
- Ascend 950PR&950DT products: Only input b supports the FRACTAL_NZ format.

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 and FLOAT32 are not supported.

## Constraints

- Empty tensors are supported. In the empty tensor scenario, bias is not supported.
- Contiguous tensors are supported. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_addmm](examples/test_aclnn_addmm_aclnninplace_addmm.cpp) | Calls the MatMulV3 operator through <br> - [aclnnAddmm&aclnnInplaceAddmm](./docs/aclnnAddmm&aclnnInplaceAddmm_en.md)<br> - [aclnnMatmul](docs/aclnnMatmul_en.md)<br> - [aclnnMatmulWeightNz](docs/aclnnMatmulWeightNz_en.md)<br> - [aclnnMm](docs/aclnnMm_en.md)<br>- [aclnnAddmmWeightNz](docs/aclnnAddmmWeightNz_en.md)<br>and other interfaces. |
