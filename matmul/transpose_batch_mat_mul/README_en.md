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

- Operator function: performs the matrix multiplication of tensor x1 and tensor x2. Only three-dimensional tensors are supported as inputs. Tensors support transposition, and the transpose sequence changes according to the passed-in sequence. permX1 represents the transpose sequence of tensor x1 and supports [0,1,2] and [1,0,2]; permX2 represents the transpose sequence of tensor x2, [0,1,2]; permY represents the transpose sequence of the matrix multiplication output matrix, and currently only [1,0,2] is supported. The sequence value 0 is the batch dimension, and the other two dimensions perform matrix multiplication. scale represents the quantization coefficient of the output matrix and can be enabled when the input is FLOAT16 and the output is INT8. For detailed constraints, see Constraints or the [aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul_en.md) invocation document.

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
    - bias is a reserved parameter and is currently not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_transpose_batch_mat_mul](examples/test_aclnn_transpose_batch_mat_mul.cpp) | Invokes the TransposeBatchMatMul operator through<br>- [aclnnTransposeBatchMatMul](docs/aclnnTransposeBatchMatMul_en.md)<br>- [aclnnTransposeBatchMatMulWeightNz](docs/aclnnTransposeBatchMatMulWeightNZ_en.md)<br>and other methods.|
| torch interface  | [torchapi_transpose_batch_mat_mul](docs/torchapi_transpose_batch_mat_mul_en.md) | Invokes the TransposeBatchMatMul operator through `cann_ops_nn.transpose_batch_mat_mul`; supports perm transpose, batch_split_factor splitting, and FLOAT16+scale INT8 quantization output; supports eager mode and TorchAir graph mode. |
