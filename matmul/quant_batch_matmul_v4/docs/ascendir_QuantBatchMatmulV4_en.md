# QuantBatchMatmulV4

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Operator function: Perform matrix multiplication in quantization scenarios (fusion computation of dequantization and matrix multiplication). The operator supports quantization scenarios such as A8W8, A8W4, MxA8W4, and A4W4, and supports accumulating a bias vector on the multiplication result. It supports quantization modes such as T-T, T-C, K-T, K-C, K-G, G-B, B-B, MX, and T-CG. For [quantization modes](../../../docs/en/context/quant_mode_introduction.md), the input and output data type combinations and constraints corresponding to different quantization modes are described in the constraints of [aclnnQuantMatmulV5](aclnnQuantMatmulV5_en.md).
- Formula (taking the K-C && K-T quantization modes as an example):

  $$
  y = x1 @ x2 * x2Scale * x1Scale + bias
  $$

  where:
  - $x1$: left matrix. The first 0 to 4 dimensions are batch dimensions (which may not exist), and the last two dimensions are $(M, K)$; when transpose_x1=true, the last two dimensions passed in for $x1$ are $(K, M)$, which are logically transposed to $(M, K)$ during computation.
  - $x2$: quantized weight right matrix. The first 0 to 4 dimensions are batch dimensions (which may not exist), and the last two dimensions are $(K, N)$; when transpose_x2=true, the last two dimensions passed in for $x2$ are $(N, K)$, which are logically transposed to $(K, N)$ during computation.
  - $x1Scale$ and $x2Scale$: scaling factors of quantization parameters. The shape varies with the quantization mode.
  - $bias$: bias vector with dimensions $(N)$, $(1, N)$, or $(B, 1, N)$.
  - $y$: output matrix. The first 0 to 4 dimensions are batch dimensions (which may not exist), and the last two dimensions are $(M, N)$. B is the batch count of the matrix multiplication, M is the number of rows of $x1$ and of $y$, N is the number of columns of $x2$ and of $y$, and K is the number of columns of $x1$ and the number of rows of $x2$.

## Ascend IR Definition

The header file of the Ascend IR definition: [quant_batch_matmul_v4_proto.h](../op_graph/quant_batch_matmul_v4_proto.h)

```cpp
REG_OP(QuantBatchMatmulV4)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1, DT_INT4, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .OP_END_FACTORY_REG(QuantBatchMatmulV4)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | Required input | Left matrix of the matrix multiplication, $x1$ in the formula. | When transpose_x1=false, the last two dimensions are (M, K); when transpose_x1=true, the last two dimensions are (K, M), which are logically transposed to (M, K) for computation. | FLOAT8_E5M2<sup>1</sup>, FLOAT8_E4M3FN<sup>1</sup>, INT8, INT4 | ND | 2~6 |
| x2 | Required input | Quantized weight right matrix of the matrix multiplication, $x2$ in the formula. | When transpose_x2=false, the last two dimensions are (K, N); when transpose_x2=true, the last two dimensions are (N, K), which are logically transposed to (K, N) for computation. In FRACTAL_NZ format, when transpose_x2=true, the shape is (k1, n1, n0, k0), where n0=16 and k0=32 (k0=64 when x2 is INT4); when transpose_x2=false, the shape is (n1, k1, k0, n0), where k0=16 and n0=32 (n0=64 when x2 is INT4). | FLOAT4_E2M1<sup>1</sup>, INT4, INT8 | ND, FRACTAL_NZ | 2~6 |
| bias | Optional input | Bias vector accumulated on the matrix multiplication result, $bias$ in the formula. | N must be the same as the N dimension of x2. | BFLOAT16, FLOAT16, FLOAT32 | ND | 1~3 |
| x1_scale | Optional input | Scaling factor of x1 quantization parameters, $x1Scale$ in the formula. | The shape varies with the quantization mode. | FLOAT16<sup>1</sup>, BFLOAT16<sup>1</sup>, FLOAT8_E8M0<sup>1</sup>, FLOAT32 | ND | 1~7 |
| x2_scale | Optional input | Scaling factor of x2 quantization parameters, $x2Scale$ in the formula. | The shape varies with the quantization mode. | FLOAT16<sup>1</sup>, BFLOAT16, FLOAT8_E8M0<sup>1</sup>, UINT64, FLOAT32 | ND | 1~7 |
| y_scale | Optional input | Scaling factor of the quantization parameters of output y. | Used in the T-CG quantization mode. The shape is (1, N). | UINT64 | ND | 2 |
| x1_offset | Optional input | Offset factor of x1 quantization parameters. | Reserved parameter. Currently not supported. | FLOAT16, BFLOAT16 | ND | 1~7 |
| x2_offset | Optional input | Offset factor of x2 quantization parameters. | The shape is (t,), where t=1 or N. N is the same as the N dimension of x2. | FLOAT16, BFLOAT16 | ND | 1~2 |
| y_offset | Optional input | Offset factor of the quantization parameters of output y. | Used in the K-G quantization mode. The shape is (N,). | FLOAT16, BFLOAT16, FLOAT32 | ND | 1 |
| x2_table | Optional input | Lookup table of x2 quantization parameters. | Reserved parameter. Currently not supported. | INT8 | ND | 2 |
| y | Required output | Computation result of the matrix multiplication, $y$ in the formula. | The batch dimension is the broadcast result of x1 and x2, and the last two dimensions are (M, N). | FLOAT16, BFLOAT16 | ND | 2~6 |
| dtype | Required attribute | Data type of output y. | Values: 1 (FLOAT16) and 27 (BFLOAT16). | Int | - | - |
| compute_type | Required attribute | Computation type. | Default value: -1 (auto). Values: -1 (auto) and 4 (f8f4). | Int | - | - |
| transpose_x1 | Required attribute | Indicates whether the left matrix x1 is transposed before the matrix multiplication. | Default value: false. When the value is true, the last two dimensions of x1 are logically transposed for computation. | Bool | - | - |
| transpose_x2 | Required attribute | Indicates whether the right matrix x2 is transposed before the matrix multiplication. | Default value: false. When the value is true, the last two dimensions of x2 are logically transposed for computation. | Bool | - | - |
| group_size | Required attribute | Quantization group size. It consists of three values: groupSizeM, groupSizeN, and groupSizeK. Each value occupies 16 bits, and the three values occupy the lower 48 bits. | Default value: -1. It takes effect only in the MX, G-B, B-B, K-G, and T-CG quantization modes. The formula is group_size = groupSizeK \| groupSizeN << 16 \| groupSizeM << 32. | Int | - | - |

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The superscript "1" in the Data type column of the preceding table indicates data types not supported by these series. The y_scale input is not supported.
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: x2_offset and y_offset are not supported currently.
<!-- end id8 -->

## Constraints

- Deterministic computation: The implementation is deterministic by default.
- Empty tensor:
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: Empty tensors are not supported.
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: Empty tensors are supported in full quantization scenarios. When x1 is an empty tensor with m=0 or x2 is an empty tensor with n=0, the output is an empty tensor (when x2 is in FRACTAL_NZ format, only an empty tensor with m=0 for x1 is supported). Empty tensors are not supported in pseudo-quantization (A8W4) scenarios.
  <!-- end id10 -->
- Contiguous tensors are supported. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) are supported only in transpose scenarios.
- For detailed constraints such as the input and output data type combinations, shape value relationships, and alignment requirements in each quantization mode, see the constraints in [aclnnQuantMatmulV5](aclnnQuantMatmulV5_en.md).

## Invocation Example

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR API | [test_geir_quant_batch_matmul_v4](../examples/arch35/test_geir_quant_batch_matmul_v4.cpp) | Construct a graph and invoke the operator through the [QuantBatchMatmulV4 operator IR](../op_graph/quant_batch_matmul_v4_proto.h). |
