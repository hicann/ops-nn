# MatMulV3

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
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Operator function: performs the general matrix multiplication (GEMM) computation of two 2D matrices, supports controlling through attributes whether the left matrix and the right matrix are transposed before participating in the computation, and supports accumulating a bias vector by column on the multiplication result, that is, the "multiply, add" operation is completed in one step.
- Formula:

  $$
  y = op(x1) @ op(x2) + bias
  $$

  where:
  - $op(x1)$: the left matrix participating in the matrix multiplication, with dimensions $(M, K)$. When transpose_x1=false, $op(x1)=x1$; when transpose_x1=true, $op(x1)=x1^T$ (x1 is passed in as $(K, M)$ and is logically transposed to $(M, K)$ during computation).
  - $op(x2)$: the right matrix participating in the matrix multiplication, with dimensions $(K, N)$. When transpose_x2=false, $op(x2)=x2$; when transpose_x2=true, $op(x2)=x2^T$ (x2 is passed in as $(N, K)$ and is logically transposed to $(K, N)$ during computation).
  - $bias$: a bias vector with dimensions $(N)$ or $(1, N)$, accumulated element-wise onto the corresponding columns of the multiplication result.
  - $y$: the output matrix, with dimensions $(M, N)$.

## Ascend IR Definition

The header file containing the Ascend IR definition is located at: [mat_mul_v3_proto.h](../op_graph/mat_mul_v3_proto.h)

```cpp
REG_OP(MatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(opImplMode, Int, 0x1)
    .OP_END_FACTORY_REG(MatMulV3)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | Mandatory input | Left matrix of the matrix multiplication, corresponding to $op(x1)$ in the formula. | When transpose_x1=false, the shape is (M, K); when transpose_x1=true, the shape is (K, M), and x1 is logically transposed to (M, K) during computation. | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2 |
| x2 | Mandatory input | Right matrix of the matrix multiplication, corresponding to $op(x2)$ in the formula. | When transpose_x2=false, the shape is (K, N); when transpose_x2=true, the shape is (N, K), and x2 is logically transposed to (K, N) during computation. In FRACTAL_NZ format, x2 is 4-dimensional: when not transposed, the layout is (n1, k1, k0, n0); when transposed, the layout is (k1, n1, n0, k0), where k0=16 and n0=16. | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2 (ND) or 4 (NZ) |
| bias | Optional input | Bias vector accumulated on the matrix multiplication result, corresponding to $bias$ in the formula. | The length must be consistent with the N dimension of x2. | FLOAT16, BFLOAT16, FLOAT32 | ND | 1~2 |
| offset_w | Optional input | Weight quantization offset matrix. | Reserved parameter, currently not supported. | INT8, INT4 | ND | 2 |
| y | Mandatory output | Computation result of the matrix multiplication, corresponding to $y$ in the formula. | - | FLOAT16, BFLOAT16, FLOAT32 | ND, FRACTAL_NZ | 2 |
| transpose_x1 | Mandatory attribute | Indicates whether the left matrix x1 is transposed before participating in the matrix multiplication. | The default value is false; when the value is true, x1 participates in the computation after logical transposition. | Bool | - | - |
| transpose_x2 | Mandatory attribute | Indicates whether the right matrix x2 is transposed before participating in the matrix multiplication. | The default value is false; when the value is true, x2 participates in the computation after logical transposition. | Bool | - | - |
| offset_x | Mandatory attribute | Quantization offset of the left matrix x1. | Reserved parameter, currently not supported. The default value is 0. | Int | - | - |
| opImplMode | Mandatory attribute | Operator implementation mode. | The default value is 0x1 (default). Supported values: 0x1 (default, default mode), 0x2 (high_performance, high-performance mode), 0x4 (high_precision, high-precision mode), 0x8 (super_performance, super-high-performance mode), 0x10 (support_of_bound_index), 0x20 (enable_float_32_execution, FLOAT32 execution), 0x40 (enable_hi_float_32_execution, HF32 execution). | Int | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: only x2 supports the FRACTAL_NZ format.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: the FRACTAL_NZ format is not supported, and bias does not support the BFLOAT16 data type.
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: only the FLOAT16 data type is supported, and x2 supports only the FRACTAL_NZ format.
<!-- end id9 -->

## Constraints

- Deterministic computation: default deterministic implementation.
- Empty tensors are supported, and bias is not supported in empty tensor scenarios.
- Contiguous tensors are supported, and [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) support only the transposition scenario.
- x1 and x2 must satisfy the data type [deduction relationship](../../../docs/en/context/deduction_relationship.md).
<!-- npu="A3,910b" id10 -->
- <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: type deduction with the two inputs being BFLOAT16 and FLOAT16, or BFLOAT16 and FLOAT32, is not supported.
<!-- end id10 -->

## Invocation Sample

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_addmm](../examples/test_aclnn_addmm_aclnninplace_addmm.cpp) | Invokes the MatMulV3 operator through<br> - [aclnnAddmm&aclnnInplaceAddmm](aclnnAddmm&aclnnInplaceAddmm_en.md)<br> - [aclnnMatmul](aclnnMatmul_en.md)<br> - [aclnnMatmulWeightNz](aclnnMatmulWeightNz_en.md)<br> - [aclnnMm](aclnnMm_en.md)<br>- [aclnnAddmmWeightNz](aclnnAddmmWeightNz_en.md)<br> and other methods. |
