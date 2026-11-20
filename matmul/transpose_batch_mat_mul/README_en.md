# TransposeBatchMatMul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: multiplies three-dimensional x1 and x2 with optional bias. permX1/permX2 specify input axis positions; permY selects (M,B,N) or, on Ascend 950PR&950DT products, (B,M,N). scale retains the existing FLOAT16-to-INT8 quantization. See [aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul_en.md) for supported combinations.

- Example:
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor equals 1, the computed output out has the shape (M, B, N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is not None, and batchSplitFactor equals 1, the computed output out has the shape (M, 1, B * N).
  - When the shape of x1 is (B, M, K), the shape of x2 is (B, K, N), scale is None, and batchSplitFactor is greater than 1, the computed output out has the shape (batchSplitFactor, M, B * N / batchSplitFactor).

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 1034px"><colgroup>
<col style="width: 120px">
<col style="width: 123px">
<col style="width: 352px">
<col style="width: 320px">
<col style="width: 125px">
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
    <td>Left matrix in the matrix multiplication.</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>Input</td>
    <td>Right matrix in the matrix multiplication.</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>Input</td>
    <td>Bias accumulated after the matrix multiplication.</td>
    <td>FLOAT32, FLOAT16, BF16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale</td>
    <td>Input</td>
    <td>Scale factor of the quantization parameters.</td>
    <td>INT64, UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>permX1</td>
    <td>Input</td>
    <td>Represents the transpose sequence of the first matrix in the matrix multiplication.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permX2</td>
    <td>Input</td>
    <td>Represents the transpose sequence of the second matrix in the matrix multiplication.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>permY</td>
    <td>Input</td>
    <td>Represents the transpose sequence of the matrix multiplication output matrix.</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>cubeMathType</td>
    <td>Input</td>
    <td>Specifies the computation logic of the Cube unit.</td>
    <td>INT8</td>
    <td>-</td>
  </tr>
  <tr>
    <td>batchSplitFactor</td>
    <td>Input</td>
    <td>Specifies the splitting size of the B dimension in the matrix multiplication output matrix.</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Computation result of the matrix multiplication.</td>
    <td>FLOAT32, FLOAT16, BF16, INT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.
- Atlas A2 products: only the input x2 supports the FRACTAL_NZ format.
- Atlas A3 products: only the input x2 supports the FRACTAL_NZ format.
- Ascend 950PR&950DT products: only the input x2 supports the FRACTAL_NZ format.

## Constraints

- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
    - Empty tensors are not supported.
    - Non-contiguous tensors are supported.
    - The value range of B is [1, 65536), and the value range of N is [1, 65536).
    - When the input shape of x1 is (B, M, K), K <= 65535; when the input shape of x1 is (M, B, K), B * K <= 65535.
    - When scale is not empty, batchSplitFactor can only equal 1, the product of B and N is less than 65536, and only the type deduction of FLOAT16 input and INT8 output is supported.
- <term>Ascend 950PR&950DT products</term>:
    - Empty tensors are not supported.
    - When scale is not empty, batchSplitFactor can only equal 1, and only the type deduction of FLOAT16 input and INT8 output is supported.
    - Optional ND bias has shape (N,) or (1,N), is shared across batches and rows, and has the same dtype as x1 or FLOAT32.
    - A inner transpose [0,2,1] and ND KBN B [1,0,2] require BMN output, no scale and batchSplitFactor=1.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_transpose_batch_mat_mul](examples/test_aclnn_transpose_batch_mat_mul.cpp) | Invokes the TransposeBatchMatMul operator through<br>- [aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul_en.md)<br>- [aclnnTransposeBatchMatMulWeightNz](docs/aclnnTransposeBatchMatMulWeightNZ_en.md)<br>and other methods.|
| torch interface  | [torchapi_transpose_batch_mat_mul](docs/torchapi_transpose_batch_mat_mul_en.md) | Invokes the TransposeBatchMatMul operator through `cann_ops_nn.transpose_batch_mat_mul`; supports perm transpose, batch_split_factor splitting, and FLOAT16+scale INT8 quantization output; supports eager mode and TorchAir graph mode. |
