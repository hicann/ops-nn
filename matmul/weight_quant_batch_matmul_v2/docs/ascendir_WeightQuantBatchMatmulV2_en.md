# WeightQuantBatchMatmulV2

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

- Operator function: performs a weight-only quantized matrix multiplication computation (a fused computation of weight dequantization and matrix multiplication), and supports quantization computation on the output. Supports quantization modes such as MX, pergroup, perchannel, and pertensor. For details, see [Quantization Modes](../../../docs/en/context/quant_mode_introduction.md). For the input shapes and constraints corresponding to different quantization modes, refer to the constraints in [aclnnWeightQuantBatchMatmulV2](aclnnWeightQuantBatchMatmulV2_en.md).
- Formula:

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  Here:
  - $x$: left matrix with dimensions $(M, K)$; when transpose_x=true, $x$ is passed in with dimensions $(K, M)$ and is logically transposed to $(M, K)$ for computation.
  - $weight$: quantized weight right matrix in the pseudo-quantization scenario, with dimensions $(K, N)$; when transpose_weight=true, $weight$ is passed in with dimensions $(N, K)$ and is logically transposed to $(K, N)$ for computation.
  - $ANTIQUANT(weight)$: weight dequantization computation, with the formula $ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale$.
  - $bias$: bias vector with dimensions $(N)$ or $(1, N)$.
  - $y$: output matrix with dimensions $(M, N)$. M is the number of rows of $x$ and of $y$, N is the number of columns of $weight$ and of $y$, and K is the number of columns of $x$ and the number of rows of $weight$.

  When the output needs to be quantized, the quantization formula is:

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

## Ascend IR Definition

The header file path of the Ascend IR definition: [weight_quant_batch_matmul_v2_proto.h](../op_graph/weight_quant_batch_matmul_v2_proto.h)

```cpp
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_INT8, DT_INT4, DT_INT32, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)
```

## Parameters

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x | Required input | Left matrix of the matrix multiplication operation, corresponding to $x$ in the formula. | When transpose_x=false, the shape is (M, K); when transpose_x=true, the shape is (K, M) and is logically transposed to (M, K) for computation. M ranges from [1, 2147483647], and K is at least 1. | FLOAT16, BFLOAT16 | ND | 2 |
| weight | Required input | Quantized weight right matrix of the matrix multiplication operation, corresponding to $weight$ in the formula. | When transpose_weight=false, the shape is (K, N); when transpose_weight=true, the shape is (N, K). When the data type is INT4 or FLOAT4_E2M1, the size of the inner axis involved in computation must be even. When the data type is INT32, the input is packed INT4 data (eight 4-bit values stored per 32 bits), and the shapes (N, K/8) and (K, N/8) are supported. | INT8, INT4, INT32, FLOAT8_E4M3FN, HIFLOAT8, FLOAT4_E2M1 | ND, FRACTAL_NZ | 2 |
| antiquant_scale | Required input | Scale factor of the dequantization parameters, corresponding to $antiquantScale$ in the formula. | In pergroup quantization mode, the shape is (⌈K / antiquant_group_size⌉, N) or (N, ⌈K / antiquant_group_size⌉); in perchannel quantization mode, the shape is (N,), (1, N), or (N, 1); in pertensor quantization mode, the shape is (1,) or (1, 1). When the data type is FLOAT16 or BFLOAT16, it must be consistent with x. | FLOAT16, BFLOAT16, UINT64, INT64, FLOAT8_E8M0 | ND | 1~2 |
| antiquant_offset | Optional input | Offset factor of the dequantization parameters, corresponding to $antiquantOffset$ in the formula. | The shape is consistent with antiquant_scale; when the data type of antiquant_scale is UINT64 or INT64, the data type of this parameter is INT32. | FLOAT16, BFLOAT16, INT32 | ND | 1~2 |
| quant_scale | Optional input | Scale factor of the output quantization parameters, corresponding to $quantScale$ in the formula. | The shapes (1,), (1, N), and (N,) are supported; this parameter is required when the data type of output y is INT8. | FLOAT32, UINT64 | ND | 1~2 |
| quant_offset | Optional input | Offset factor of the output quantization parameters, corresponding to $quantOffset$ in the formula. | The shape is consistent with quant_scale. | FLOAT32 | ND | 1~2 |
| bias | Optional input | Bias vector added to the matrix multiplication result, corresponding to $bias$ in the formula. | The shapes (N,) and (1, N) are supported, and N must be consistent with the N dimension of weight. | FLOAT16, FLOAT32, BFLOAT16 | ND | 1~2 |
| y | Required output | Computation result of the matrix multiplication operation, corresponding to $y$ in the formula. | The shape is (M, N); when quant_scale exists, the data type is INT8, otherwise the data type is consistent with x. | FLOAT16, BFLOAT16, INT8 | ND | 2 |
| transpose_x | Required attribute | Indicates whether the left matrix x is transposed before matrix multiplication. | The default value is false; when set to true, x participates in the computation after logical transposition. | Bool | - | - |
| transpose_weight | Required attribute | Indicates whether the right matrix weight is transposed before matrix multiplication. | The default value is false; when set to true, weight participates in the computation after logical transposition. | Bool | - | - |
| antiquant_group_size | Required attribute | Group size for the dequantization computation on weight in MX and pergroup quantization modes; it describes the size along the Reduce direction of the amount of data to be dequantized corresponding to one group of dequantization parameters. | The default value is 0, indicating that the pergroup quantization mode is not used. The value range is [0, K-1], and the value must be a multiple of 32; it must be 32 when the data type of antiquant_scale is FLOAT8_E8M0. | Int | - | - |
| dtype | Required attribute | Data type of output y. | The default value is -1; when quant_scale exists in the input, the output is INT8, otherwise the output is consistent with the data type of x. Supported values are 1 (FLOAT16), 2 (INT8), and 27 (BFLOAT16). | Int | - | - |
| inner_precise | Required attribute | Computation mode. | The default value is 0; supported values are 0 (high-precision mode) and 1 (high-performance mode). | Int | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - x: Transposition is not supported (transpose_x supports only false), and only contiguous tensors are supported.
  - weight: When the data type is INT32, it serves as the carrier of packed INT4 data; when the data type is FLOAT8_E4M3FN or HIFLOAT8, only the ND format is supported; when the data format is FRACTAL_NZ, transpose_weight supports only false.
  - quant_scale and quant_offset: Not supported for now.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: weight supports only the INT8, INT4, and INT32 data types; antiquant_scale supports only the FLOAT16, BFLOAT16, UINT64, and INT64 data types.
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas inference products</term>: x supports only the FLOAT16 data type, weight supports only the INT8 data type, y supports only the FLOAT16 data type, and antiquant_scale supports only the FLOAT16 data type.
<!-- end id9 -->

## Constraints

- Deterministic computation:
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: Default deterministic implementation.
  <!-- end id10 -->
  <!-- npu="A3,910b,310p" id11 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>: Default non-deterministic implementation; determinism can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id11 -->
- Empty tensors are not supported.
- Contiguous tensors are supported; [non-contiguous tensors](../../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.
- The Reduce dimension K of x and weight must be equal in size.
- For detailed constraints in each quantization mode, such as data type combinations of inputs and outputs, shape value relationships, and alignment requirements, refer to the constraints in [aclnnWeightQuantBatchMatmulV2](aclnnWeightQuantBatchMatmulV2_en.md).

## Invocation Example

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR interface | [test_geir_weight_quant_batch_matmul_v2](../examples/arch35/test_geir_weight_quant_batch_matmul_v2.cpp) | Constructs the graph through the [WeightQuantBatchMatmulV2 operator IR](../op_graph/weight_quant_batch_matmul_v2_proto.h) and invokes the operator. |
