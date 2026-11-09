# QuantBatchMatmulV3

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

- <term>Atlas 200I/500 A2 inference products</term>: Not Supported

<!-- end id4 -->

<!-- npu="310p" id5 -->

- <term>Atlas inference products</term>: Supported

<!-- end id5 -->

<!-- npu="910" id6 -->

- <term>Atlas training products</term>: Not Supported

<!-- end id6 -->

- Kirin X90 processor series products: Supported
- Kirin 9030 processor series products: Supported

## Function

- Operator function: Performs quantized matrix multiplication with result scaling, bias accumulation, and output type conversion. Supports [quantization modes](../../../docs/en/context/quant_mode_introduction.md) such as T-T, T-C, K-T, K-C, G-B, B-B, and MX. Supported modes and data type combinations depend on the product.
- Formulas (using quantization without grouping along K as an example):
  - Without bias:

    $$
    y = (op(x1) @ op(x2)) * scale * pertokenScale + offset
    $$

  - With INT32 bias, accumulate before scaling:

    $$
    y = (op(x1) @ op(x2) + bias) * scale * pertokenScale + offset
    $$

  - With BFLOAT16, FLOAT16, or FLOAT32 bias, accumulate after scaling (offset is not supported in this case):

    $$
    y = (op(x1) @ op(x2)) * scale * pertokenScale + bias
    $$

  Here, $pertokenScale$ corresponds to input pertoken_scale and is treated as 1 when omitted. An omitted offset is treated as 0. offset is used only in supported INT8-output combinations and cannot be used together with pertoken_scale. The result is converted to the type specified by dtype. UINT64/INT64 scale stores encoded quantization parameters; its integer value cannot be substituted directly into the formulas.

  For INT32 output, the operator directly returns the integer accumulation result $y = op(x1) @ op(x2) + bias$, with optional bias, without the scaling above. scale remains a required IR input.

  G-B, B-B, and MX modes group data along K. Each group's matrix product is multiplied by the corresponding left and right scales before the group results are summed. Group scales cannot be moved after the complete matrix multiplication.

  where:
  - $op(x1)$ and $op(x2)$ are the matrices logically transposed according to transpose_x1 and transpose_x2. Their last two computation dimensions are $(M, K)$ and $(K, N)$. false leaves the input unchanged; true swaps its last two dimensions, in which case the input shape is $(K, M)$ or $(N, K)$, respectively.
  - The last two dimensions of $y$ are $(M, N)$, where M and N are the result matrix dimensions and K is the reduction dimension. Inputs and output may have 0–4 leading batch dimensions. The output uses the broadcast batch shape of the two inputs.

## Ascend IR Definition

Header containing the Ascend IR definition: [quant_batch_matmul_v3_proto.h](../op_graph/quant_batch_matmul_v3_proto.h)

```cpp
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)
```

## Parameters

The table lists types declared by the IR; arbitrary combinations are not supported. Valid combinations depend on the product and quantization mode. DT_FLOAT corresponds to FLOAT32, and DT_BF16 to BFLOAT16. The dimension column gives the logical rank. FRACTAL_NZ storage shapes include additional block axes and must not be passed as ND logical shapes. Logical shapes for INT4 and FLOAT4_E2M1 count elements, not packed bytes.

| Parameter name | Input/Attribute/Output | Description | Usage instructions | Data type | Data format | Dimensions (shape) |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | Required input | Left matrix, corresponding to $op(x1)$ in the formulas. | The last two logical dimensions are (M, K) when transpose_x1 = false and (K, M) when true. INT4 supports only two dimensions with transpose_x1 = false. FRACTAL_NZ storage shapes have 4–8 dimensions. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND, FRACTAL_NZ (product dependent) | 2–6 |
| x2 | Required input | Right matrix, corresponding to $op(x2)$ in the formulas. | The last two logical dimensions are (K, N) when transpose_x2 = false and (N, K) when true. INT4 supports only two dimensions. FRACTAL_NZ storage shapes have 4–8 dimensions and must use a layout supported by the product. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND, FRACTAL_NZ (product dependent) | 2–6 |
| scale | Required input | Right-matrix quantization scale, corresponding to $scale$ in the formulas. | Shape (1,) or (N,) for per-tensor/per-channel quantization. For grouped modes, see the table below. UINT64/INT64 stores encoded quantization parameters. | UINT64, FLOAT32, INT64, BFLOAT16, FLOAT8_E8M0 | ND | 1–7 |
| offset | Optional input | Output quantization offset, corresponding to $offset$ in the formulas. | Shape (1,) or (N,). Supported only in the corresponding INT8-output combinations. Treated as 0 when omitted. | FLOAT32 | ND | 1 |
| bias | Optional input | Matrix-product accumulation bias, corresponding to $bias$ in the formulas. | Shape (N,) or (B, 1, N), where B is the output batch count. Outputs of rank 2, 4, 5, or 6 require one-dimensional bias. Rank-3 outputs support one- or three-dimensional bias. | INT32, BFLOAT16, FLOAT16, FLOAT32 | ND | 1 or 3 |
| pertoken_scale | Optional input | Left-matrix quantization scale, corresponding to $pertokenScale$ in the formulas. | For non-grouped modes, INT8 and INT4 inputs require shape (M,); FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 inputs support (1,) or (M,), subject to the WeightNZ restrictions below. For grouped modes, see the table below. Treated as 1 when omitted; required for G-B, B-B, and MX. | FLOAT32, FLOAT8_E8M0 | ND | 1–7 |
| y | Required output | Matrix multiplication result, corresponding to $y$ in the formulas. | Uses the broadcast batch shape of x1 and x2, with last two logical dimensions (M, N). dtype specifies its data type. Per-token quantization on Atlas inference products uses FRACTAL_NZ. | FLOAT16, INT8, BFLOAT16, INT32, FLOAT32 | ND, FRACTAL_NZ (product dependent) | 2–6 |
| dtype | Required attribute | Specifies the data type of output y. | Must be set explicitly; the IR has no default. Values: 0 (FLOAT32), 1 (FLOAT16), 2 (INT8), 3 (INT32), and 27 (BFLOAT16), subject to product and input-combination restrictions. | Int | - | - |
| transpose_x1 | Optional attribute | Whether to transpose left matrix x1 before multiplication. | Defaults to false. true logically transposes the last two dimensions. | Bool | - | - |
| transpose_x2 | Optional attribute | Whether to transpose right matrix x2 before multiplication. | Defaults to false. true logically transposes the last two dimensions. | Bool | - | - |
| group_size | Optional attribute | Group size for grouped quantization. | Defaults to 0. The low 16 bits store group_size_k, bits 16–31 store group_size_n, and bits 32–47 store group_size_m: group_size = group_size_k \| (group_size_n << 16) \| (group_size_m << 32). | Int | - | - |

### Shapes for Grouped Quantization

<!-- npu="950" id11 -->

The following table describes G-B, B-B, and MX grouped-quantization shapes on <term>Ascend 950PR&950DT products</term>. batch_a and batch_b denote the optional 0–4 leading dimensions of x1 and x2. The batch dimensions of pertoken_scale match batch_a, and those of scale match batch_b. batch_a and batch_b may differ but must be broadcast-compatible. scale and pertoken_scale use ND format; x1 and x2 columns show logical shapes. The G-B and B-B rows assume neither input is transposed. The MX rows describe shapes for the transpose attributes, not support for every type/format combination. Under the current IR contract, FLOAT4_E2M1 WeightNZ supports only the first two MX rows, where x1 is not transposed, subject to the WeightNZ constraints below. The last two rows with transposed x1 do not apply to that case.

| Quantization mode | x1 shape | x2 shape | pertoken_scale shape | scale shape | Resulting [group_size_m, group_size_n, group_size_k] |
| -------- | -------- | -------- | -------------------- | ----------- | ------------------------------------------ |
| G-B | (batch_a, M, K) | (batch_b, K, N) | (batch_a, M, ⌈K/128⌉) | (batch_b, ⌈K/128⌉, ⌈N/128⌉) | [1, 128, 128] |
| B-B | (batch_a, M, K) | (batch_b, K, N) | (batch_a, ⌈M/128⌉, ⌈K/128⌉) | (batch_b, ⌈K/128⌉, ⌈N/128⌉) | [128, 128, 128] |
| MX | (batch_a, M, K) | (batch_b, K, N) | (batch_a, M, ⌈K/64⌉, 2) | (batch_b, ⌈K/64⌉, N, 2) | [1, 1, 32] |
| MX | (batch_a, M, K) | (batch_b, N, K) | (batch_a, M, ⌈K/64⌉, 2) | (batch_b, N, ⌈K/64⌉, 2) | [1, 1, 32] |
| MX | (batch_a, K, M) | (batch_b, K, N) | (batch_a, ⌈K/64⌉, M, 2) | (batch_b, ⌈K/64⌉, N, 2) | [1, 1, 32] |
| MX | (batch_a, K, M) | (batch_b, N, K) | (batch_a, ⌈K/64⌉, M, 2) | (batch_b, N, ⌈K/64⌉, 2) | [1, 1, 32] |

- scale and pertoken_scale use FLOAT32 for G-B and B-B, and FLOAT8_E8M0 for MX.
- Zero group-size components are inferred from input shapes. The corresponding dimensions must be divisible during inference, and the inferred values must match the combinations above. Other quantization modes require group_size=0.
- If a grouping dimension is not divisible by its group size, explicitly specify the group size to avoid inference failure with 0. For example, use `(int64_t{1} << 32) | (int64_t{1} << 16) | 32` for MX. The last dimension of size 2 holds scales for two adjacent groups of 32 elements along K; MX group_size_k must not be set to 64.

<!-- end id11 -->

### Product-Specific Usage

<!-- npu="A3,910b" id7 -->

#### <term>Atlas A3 products</term>, <term>Atlas A2 products</term>

x1 and x2 support only INT8 and INT4. scale does not support FLOAT8_E8M0; pertoken_scale supports only FLOAT32; y does not support FLOAT32. If x1 uses FRACTAL_NZ, x2 must also use FRACTAL_NZ, x1 cannot be INT4, and y cannot be INT8. The last logical dimension of x1 and x2 must not exceed 65535.

<!-- end id7 -->

<!-- npu="310p" id8 -->

#### <term>Atlas inference products</term>

x1 and x2 support only INT8. scale supports UINT64 and INT64, and also FLOAT32 for per-token quantization. bias supports INT32, and also FLOAT32 for per-token quantization. y supports FLOAT16 and INT8, but only FLOAT16 for per-token quantization. The x1/x2/y format combination is ND/FRACTAL_NZ/ND without per-token quantization and FRACTAL_NZ/FRACTAL_NZ/FRACTAL_NZ with per-token quantization.

<!-- end id8 -->

<!-- npu="950" id9 -->

#### <term>Ascend 950PR&950DT products</term>

For the ND-scale cases covered here, only x2 supports FRACTAL_NZ; all other inputs and outputs use ND. For WeightNZ, x1 and x2 must have the same type: INT8, HIFLOAT8, FLOAT8_E4M3FN, or FLOAT4_E2M1. INT4 and FLOAT8_E5M2 are not supported. Neither K nor N can be 1.

- For non-MX HIFLOAT8 and FLOAT8_E4M3FN: if pertoken_scale is omitted, scale uses UINT64/INT64. If pertoken_scale is provided, both scales use FLOAT32 and transpose_x1 must be false. In the latter case, non-grouped quantization supports only K-C and K-T, with pertoken_scale shape (M,). When M > 1, shape (1,) cannot be used to implement T-C or T-T.
- For MX with FLOAT8_E4M3FN or FLOAT4_E2M1: scale and pertoken_scale must both be provided as FLOAT8_E8M0, with shapes matching an applicable transpose combination in the MX table above.
- For FLOAT4_E2M1 WeightNZ: the current [IR prototype](../op_graph/quant_batch_matmul_v3_proto.h) requires transpose_x1=false. N must be greater than 2 when transpose_x2=false, and greater than 1 when transpose_x2=true. The MX requirements below for even inner dimensions and K > 2 also apply.

<!-- end id9 -->

#### Kirin X90/Kirin 9030 processor series products

x1 and x2 support only INT8. x1 supports only ND, and x2 only FRACTAL_NZ. scale supports only UINT64 and INT64; bias only INT32; pertoken_scale only FLOAT32; and y only FLOAT16 and INT8.

## Constraints

- The computation K dimensions of x1 and x2 must match, and their batch dimensions must satisfy the [broadcasting rules](../../../docs/en/context/broadcast_relationship.md).
- G-B and B-B modes do not support bias.
- INT4 inputs support only two dimensions, require transpose_x1=false, and must have an even last dimension.
- In MX mode, the inner dimension of a FLOAT4_E2M1 input (the last dimension of its supplied logical shape) must be even, and K must be greater than 2. FRACTAL_NZ x2 must also satisfy the WeightNZ constraints above.
- Empty tensor:
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT products</term>: With ND x2, x1 M = 0 or x2 N = 0 is supported and produces an empty tensor. With FRACTAL_NZ x2, x1 M = 0 is supported and produces an empty tensor.
  <!-- end id10 -->
  - Inputs with a zero-sized dimension are not supported in other cases.
- Deterministic computation:
  <!-- npu="A3,910b,310p,950" id12 -->
  - <term>Atlas A3 products</term>, <term>Atlas A2 products</term>, <term>Atlas inference products</term>, and <term>Ascend 950PR&950DT products</term>: The implementation is deterministic by default.
  <!-- end id12 -->
- Batch consistency:
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: B-B quantization does not support [batch consistency](../../../docs/en/context/batch_consistency.md). Default determinism does not guarantee identical results across different batch combinations.
  <!-- end id13 -->
- Contiguous tensors are supported. transpose_x1 and transpose_x2 specify input transposition. FRACTAL_NZ inputs must use the required storage layout; changing only the format flag without converting the data is insufficient.
- This document describes GEIR graph construction with ND scale. For direct ACLNN and ACLNN Graph invocation with ScaleNZ, see [aclnnQuantMatmulWeightNz](aclnnQuantMatmulWeightNz_en.md#scalenz-constraints).
- For GEIR graph construction, prepare inputs according to the IR requirements; ACLNN preprocessing is not available. For example, INT8 inputs with FLOAT16 output and no pertoken_scale require encoded UINT64/INT64 scale, not a raw FLOAT32 scale.
- For additional type combinations, shapes, and alignment constraints, see the corresponding full-quantization cases in [aclnnQuantMatmulV4](aclnnQuantMatmulV4_en.md), [aclnnQuantMatmulWeightNz](aclnnQuantMatmulWeightNz_en.md), and [aclnnQuantMatmulV5](../../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5_en.md). Those ACLNN interfaces also include input conversions, interface-level checks, and cases implemented by other operators. Their complete type lists, restrictions, and invocation capabilities do not apply directly to this IR.

## Example

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| GEIR interface | [test_geir_quant_batch_matmul_v3](../examples/test_geir_quant_batch_matmul_v3.cpp) | Builds a graph and invokes the operator through the [QuantBatchMatmulV3 operator IR](../op_graph/quant_batch_matmul_v3_proto.h). |

The example uses INT8 inputs, FLOAT32 scale and pertoken_scale, and FLOAT16 output, and checks the result on the host. Run it on Atlas A2 products, Atlas A3 products, or Ascend 950PR&950DT products.

For environment setup and build instructions, see [GE Graph Mode in the Operator Invocation Guide](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式).
