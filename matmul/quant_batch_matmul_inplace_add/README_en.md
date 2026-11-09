# QuantBatchMatmulInplaceAdd

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Fuses quantized matrix multiplication and in-place addition, accumulating the matrix product into an existing y. This supports gradient accumulation during micro-batch training.

- Formulas: Supports MX (Microscaling) quantization and HIFLOAT8 T-T (per-tensor–per-tensor) quantization. See [Introduction to Quantization Modes](../../docs/en/context/quant_mode_introduction.md).

  In the following formulas, $X_1$ and $X_2$ are the matrices interpreted according to the transpose attributes, with shapes $(M,K)$ and $(K,N)$. M and N are the result matrix dimensions, and K is the reduction dimension. $Y_{\mathrm{in}}$ and $Y_{\mathrm{out}}$ denote y before and after in-place accumulation.

  - MX quantization:

    $$
    Y_{\mathrm{out},m,n}=Y_{\mathrm{in},m,n}+
    \sum_{j=0}^{L-1}\left[
    \left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}X_{2,jg_K+t,n}\right)
    S_{1,m,j}S_{2,j,n}\right]
    $$

    Here, $g_K=32$, $L=\lceil K/g_K\rceil$, and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ are the input scales for the corresponding K group. The last group may contain fewer than 32 elements.

  - HIFLOAT8 T-T quantization:

    $$
    Y_{\mathrm{out},m,n}=Y_{\mathrm{in},m,n}+
    \left(\sum_{k=0}^{K-1}X_{1,m,k}X_{2,k,n}\right)S_1S_2
    $$

    Here, $S_1$ and $S_2$ correspond to x1_scale and x2_scale, respectively, and both have shape $(1,)$.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| x1 | Input | Left matrix, corresponding to X₁ in the formulas. | FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1 | ND |
| x2 | Input | Right matrix, corresponding to X₂ in the formulas. | FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8, FLOAT4_E2M1 | ND |
| x2_scale | Input | Quantization scale for x2, corresponding to S₂ in the formulas. Uses FLOAT8_E8M0 for MX quantization and FLOAT32 for HIFLOAT8 T-T quantization. | FLOAT8_E8M0, FLOAT32 | ND |
| y | Input | Existing result used for in-place accumulation, corresponding to Y_in in the formulas. | FLOAT32 | ND |
| x1_scale | Optional input | Quantization scale for x1, corresponding to S₁ in the formulas. Required for both currently supported modes, MX and HIFLOAT8 T-T. Its data type must match x2_scale. | FLOAT8_E8M0, FLOAT32 | ND |
| y | Output | Result after in-place accumulation, corresponding to Y_out in the formulas. Shares storage with input y. | FLOAT32 | ND |
| transpose_x1 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x1: false disables transposition and true enables it.</li><li>Defaults to false; must be true for currently supported computations.</li></ul> | BOOL | - |
| transpose_x2 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x2: false disables transposition and true enables it.</li><li>Defaults to false; must remain false for currently supported computations.</li></ul> | BOOL | - |
| group_size | Optional attribute | <ul><li>Encodes quantization group sizes along M, N, and K, with 16 bits per component. For supported combinations, see <a href="#constraints">Constraints</a>.</li><li>Defaults to 0.</li></ul> | INT64 | - |

## Constraints

- Only transpose_x1=true and transpose_x2=false are currently supported. The logical input shapes of x1 and x2 are `(K, M)` and `(K, N)`, respectively, and y has shape `(M, N)`.
- group_size is encoded as `group_size_k | (group_size_n << 16) | (group_size_m << 32)`. A zero component is inferred from the corresponding input and scale shapes. The inferred sizes must satisfy the selected quantization mode.
- MX quantization:
  - x1 and x2 may each be FLOAT8_E4M3FN or FLOAT8_E5M2, or both may be FLOAT4_E2M1. Mixed FP4/FP8 inputs and FLOAT4_E1M2 are not supported.
  - x1_scale and x2_scale are FLOAT8_E8M0, with shapes `(ceil(K/64), M, 2)` and `(ceil(K/64), N, 2)`, respectively, where ceil rounds upward.
  - The resulting group sizes must be `[group_size_m, group_size_n, group_size_k]=[1, 1, 32]`. Set group_size=4295032864 explicitly, or set it to 32 to infer the M and N group sizes from the shapes.
  - For MXFP4, M and N must be nonnegative even numbers. x1 and x2 use packed 4-bit storage, with two FLOAT4_E2M1 elements per byte. For contiguous inputs, the byte-storage shapes are `(K, M/2)` and `(K, N/2)`, respectively.
  - For MXFP4, when M, N, and K are all positive, both $M\times K>2$ and $N\times K>2$ must also hold.
- HIFLOAT8 T-T quantization:
  - x1 and x2 are HIFLOAT8. x1_scale and x2_scale are FLOAT32 and both have shape `(1,)`.
  - group_size must be 0.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [MXFP8 example](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp) | Calls [aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd_en.md) to demonstrate MXFP8 quantization. |
| aclnn API | [MXFP4 example](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp4.cpp) | Calls [aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd_en.md) with packed FLOAT4_E2M1 inputs. |
| aclnn API | [HIFLOAT8 T-T example](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_TT.cpp) | Calls [aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd_en.md) to demonstrate per-tensor quantization. |
| GE graph mode | - | Builds a graph using the [QuantBatchMatmulInplaceAdd operator IR](op_graph/quant_batch_matmul_inplace_add_proto.h). |
