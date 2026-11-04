# BatchMatMulV3

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Operator function: Performs matrix multiplication with the batch dimension. The inputs support 2 to 6 dimensions. The leading dimensions are batch dimensions (broadcasting of batch dimensions is supported), the last two dimensions perform matrix multiplication, and accumulating a bias vector on the multiplication result is supported.
- Formula:

  $$
  y = x1 @ x2 + bias
  $$

  where:
  - $x1$: the left matrix, whose last two dimensions are $(M, K)$; when adj_x1=true, the last two dimensions of the passed-in $x1$ are $(K, M)$, which is logically transposed to $(M, K)$ during computation.
  - $x2$: the right matrix, whose last two dimensions are $(K, N)$; when adj_x2=true, the last two dimensions of the passed-in $x2$ are $(N, K)$, which is logically transposed to $(K, N)$ during computation.
  - $bias$: a bias vector with the dimensions $(N)$, $(1, N)$, or $(B, 1, N)$, which is added element-wise to the corresponding columns of the multiplication result.
  - $y$: the output matrix, whose last two dimensions are $(M, N)$ and whose batch dimension is the result of broadcasting $x1$ and $x2$. For example, when the input dimensions of $x1$ and $x2$ are $(B, M, K)$ and $(B, K, N)$ respectively, the dimensions of $y$ are $(B, M, N)$. B is the number of batches of the matrix multiplication, M is the number of rows of $x1$ and of $y$, N is the number of columns of $x2$ and of $y$, and K is the number of columns of $x1$ and the number of rows of $x2$.

## Ascend IR Definition

Header file path of the Ascend IR definition: [batch_mat_mul_v3_proto.h](../op_graph/batch_mat_mul_v3_proto.h)

```cpp
REG_OP(BatchMatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV3)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | Required input | Left matrix of the matrix multiplication, which is $x1$ in the formula. | When adj_x1=false, the last two dimensions are (M, K); when adj_x1=true, the last two dimensions are (K, M), which is logically transposed to (M, K) to participate in the computation. | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2~6 |
| x2 | Required input | Right matrix of the matrix multiplication, which is $x2$ in the formula. | When adj_x2=false, the last two dimensions are (K, N); when adj_x2=true, the last two dimensions are (N, K), which is logically transposed to (K, N) to participate in the computation. The reduce dimension K must be the same as that of x1. | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2~6 |
| bias | Optional input | Bias vector accumulated on the matrix multiplication result, which is $bias$ in the formula. | N must be the same as the N dimension of x2. | FLOAT16, BFLOAT16, FLOAT32 | ND | 1~3 |
| offset_w | Optional input | Weight quantization offset matrix. | Reserved parameter, not supported currently. | INT8, INT4 | ND | 2 |
| y | Required output | Computation result of the matrix multiplication, which is $y$ in the formula. | The batch dimension is the result of broadcasting x1 and x2. | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2~6 |
| adj_x1 | Required attribute | Specifies whether the left matrix x1 is transposed before the matrix multiplication. | Default value: false. When the value is true, the last two dimensions of x1 are logically transposed to participate in the computation. | Bool | - | - |
| adj_x2 | Required attribute | Specifies whether the right matrix x2 is transposed before the matrix multiplication. | Default value: false. When the value is true, the last two dimensions of x2 are logically transposed to participate in the computation. | Bool | - | - |
| offset_x | Required attribute | Quantization offset of the left matrix x1. | Reserved parameter, not supported currently. Default value: 0. | Int | - | - |
| enable_hf32 | Required attribute | Specifies whether to perform the computation in HFLOAT32 (high-precision floating-point 32-bit) mode. | Default value: false. | Bool | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: Only x2 supports the FRACTAL_NZ format.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: bias does not support the BFLOAT16 data type.
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: Only the FLOAT16 data type is supported, and x2 supports only the FRACTAL_NZ format.
<!-- end id9 -->

## Constraints

- Deterministic computation: deterministic implementation by default.
- Empty tensors are supported, but bias is not supported in the empty tensor scenario.
- Contiguous tensors are supported, and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.
- The batch dimensions of x1 and x2 must satisfy the [broadcast relationship](../../../docs/en/context/broadcast_relationship.md), and the reduce dimension K must be of the same size.

## Invocation Sample

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_batchmatmul](../examples/test_aclnn_batchmatmul.cpp) | Invoke the BatchMatMulV3 operator through<br>[aclnnAddbmm&aclnnInplaceAddbmm](aclnnAddbmm&aclnnInplaceAddbmm_en.md)<br>[aclnnBaddbmm&aclnnInplaceBaddbmm](aclnnBaddbmm&aclnnInplaceBaddbmm_en.md)<br>[aclnnBatchMatMul](aclnnBatchMatMul_en.md)<br>[aclnnBatchMatMulWeightNz](aclnnBatchMatMulWeightNz_en.md)<br>and other APIs. |
