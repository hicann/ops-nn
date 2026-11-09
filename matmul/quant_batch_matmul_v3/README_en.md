# QuantBatchMatmulV3

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | √ |

## Function

- Operator function: Performs quantized matrix multiplication with scaling, bias accumulation, and output type conversion. The logical input rank is 2–6. Supported per-tensor, per-token, per-channel, and grouped modes depend on the product and data type. See [Introduction to Quantization Modes](../../docs/en/context/quant_mode_introduction.md).

- Version relationship: QuantBatchMatmulV3 is the operator name; ACLNN interface versions are defined separately. [aclnnQuantMatmulV4](docs/aclnnQuantMatmulV4_en.md) adds left-matrix pertoken_scale compared with [aclnnQuantMatmulV3](docs/aclnnQuantMatmulV3_en.md). [aclnnQuantMatmulV5](../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5_en.md) provides additional quantization parameters and dispatches to this operator or [QuantBatchMatmulV4](../quant_batch_matmul_v4/README_en.md) according to the input combination.

- Formulas: Batch indices are omitted below. $X_1$ and $X_2$ are the matrices interpreted according to transpose_x1 and transpose_x2, with shapes $(M,K)$ and $(K,N)$. M and N are the result matrix dimensions, and K is the reduction dimension. $S_1$ and $S_2$ correspond to pertoken_scale and scale. Omitted pertoken_scale, bias, and offset are treated as 1, 0, and 0, respectively.

  - Non-grouped quantization without bias:

    $$
    y=(X_1@X_2)\odot S_2\odot S_1+offset
    $$

  - INT8 or INT4 inputs with INT32 bias:

    $$
    y=(X_1@X_2+bias)\odot S_2\odot S_1+offset
    $$

  - INT8 or INT4 inputs with BFLOAT16, FLOAT16, or FLOAT32 bias:

    $$
    y=(X_1@X_2)\odot S_2\odot S_1+bias
    $$

  - Non-grouped quantization with FLOAT8_E4M3FN, FLOAT8_E5M2, or HIFLOAT8 inputs and FLOAT32 bias:

    $$
    y=(X_1@X_2+bias)\odot S_2\odot S_1
    $$

  - INT32 output: Returns the integer accumulation result directly, without scaling.

    $$
    y=X_1@X_2+bias
    $$

  - MX (Microscaling), G-B, and B-B grouped quantization: Compute by K groups and sum the group results.

    $$
    y_{m,n}=\sum_{j=0}^{L-1}\left[
    \left(\sum_{t=0}^{g_j-1}X_{1,m,jg_K+t}X_{2,jg_K+t,n}\right)
    S_{1,m,j}S_{2,j,n}\right]+bias_n
    $$

    Here, $g_K$ is the group size along K, $L=\lceil K/g_K\rceil$, and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ are the scales broadcast according to the quantization groups. G-B and B-B do not support bias, so $bias_n=0$ in the formula.

  Here, $@$ denotes matrix multiplication, and $\odot$ denotes elementwise multiplication with broadcasting according to the quantization-parameter shapes. offset is used only in supported INT8-output combinations and cannot be combined with pertoken_scale. These formulas describe the quantized computation; the result is also converted to the type specified by dtype.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| x1 | Input | Left matrix, corresponding to X₁ in the formulas. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND, FRACTAL_NZ |
| x2 | Input | Right matrix, corresponding to X₂ in the formulas. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND, FRACTAL_NZ |
| scale | Input | Right-matrix quantization scale. UINT64 and INT64 store encoded quantization parameters; use the decoded scale in the formulas. | UINT64, FLOAT32, INT64, BFLOAT16, FLOAT8_E8M0 | ND, FRACTAL_NZ |
| offset | Optional input | Output quantization offset. Treated as 0 when omitted; used only in supported INT8-output combinations. | FLOAT32 | ND |
| bias | Optional input | Matrix-product accumulation bias. Omitted bias adds nothing. Its accumulation position depends on the input and bias data types. | INT32, BFLOAT16, FLOAT16, FLOAT32 | ND |
| pertoken_scale | Optional input | Left-matrix quantization scale. Treated as 1 when omitted. The corresponding scales are required for per-token and grouped quantization. | FLOAT32, FLOAT8_E8M0 | ND |
| y | Output | Result after matrix multiplication and quantization, with data type specified by dtype. | FLOAT16, INT8, BFLOAT16, INT32, FLOAT32 | ND, FRACTAL_NZ |
| dtype | Attribute | Output y data type: 0 (FLOAT32), 1 (FLOAT16), 2 (INT8), 3 (INT32), or 27 (BFLOAT16), subject to product and input-combination restrictions. | INT64 | - |
| transpose_x1 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x1: false disables transposition and true enables it.</li><li>Defaults to false.</li></ul> | BOOL | - |
| transpose_x2 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x2: false disables transposition and true enables it.</li><li>Defaults to false.</li></ul> | BOOL | - |
| group_size | Optional attribute | <ul><li>Encodes quantization group sizes along M, N, and K, with 16 bits per component. For supported combinations, see <a href="#constraints">Constraints</a>.</li><li>Defaults to 0.</li></ul> | INT64 | - |

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: x1 and x2 do not support HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, or FLOAT4_E2M1. scale does not support FLOAT8_E8M0; pertoken_scale supports only FLOAT32; y does not support FLOAT32.
- <term>Atlas inference products</term>: x1 and x2 support only INT8. scale supports UINT64 and INT64, and also FLOAT32 for per-token quantization. bias supports INT32, and also FLOAT32 for per-token quantization. pertoken_scale supports only FLOAT32. y supports FLOAT16 and INT8, but only FLOAT16 for per-token quantization. y uses FRACTAL_NZ for per-token quantization and ND otherwise.
- <term>Kirin X90 processor products</term>, <term>Kirin 9030 processor products</term>: x1 and x2 support only INT8; x1 uses ND and x2 uses FRACTAL_NZ. scale supports only UINT64 and INT64; bias only INT32; pertoken_scale only FLOAT32; y only FLOAT16 and INT8.
- <term>Ascend 950PR&950DT products</term>: FRACTAL_NZ scale is used only for ScaleNZ in full MX quantization. x2 must also use FRACTAL_NZ, and x1 and x2 must share the same type, FLOAT8_E4M3FN or FLOAT4_E2M1. For layout and conversion requirements, see [ScaleNZ Constraints](docs/aclnnQuantMatmulWeightNz_en.md#scalenz-constraints). scale uses ND on other products.

## Constraints

- The input K dimensions must match, and batch dimensions must satisfy the [broadcasting rules](../../docs/en/context/broadcast_relationship.md). For shapes and product-specific restrictions, see the [Operator IR Description](docs/ascendir_QuantBatchMatmulV3_en.md).
- INT4 inputs support only two dimensions, require transpose_x1=false, and must have an even innermost dimension.
- group_size is encoded as `group_size_k | (group_size_n << 16) | (group_size_m << 32)`. Zero components are inferred from input shapes, with divisibility required for the corresponding dimensions. The resulting sizes for MX, G-B, and B-B must be `[1, 1, 32]`, `[1, 128, 128]`, and `[128, 128, 128]`, respectively. Other modes use group_size=0.
- MX requires both pertoken_scale and scale as FLOAT8_E8M0. FLOAT4_E2M1 inputs must have an even innermost dimension, and K must be greater than 2.
- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>: The last logical dimension of each input matrix must not exceed 65535.
- <term>Ascend 950PR&950DT products</term>: ND x2 supports M=0 or N=0 and produces an empty tensor. FRACTAL_NZ x2 supports M=0 and produces an empty tensor. Other cases do not support inputs with zero-sized dimensions.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_quant_batch_matmul_v3](examples/test_geir_quant_batch_matmul_v3.cpp) | Builds a graph using the [QuantBatchMatmulV3 operator IR](docs/ascendir_QuantBatchMatmulV3_en.md). |
| aclnn API | [test_aclnn_quant_matmul_v3_at](examples/test_aclnn_quant_matmul_v3_at.cpp) | Invokes [aclnnQuantMatmulV3](docs/aclnnQuantMatmulV3_en.md). |
| aclnn API | [test_aclnn_quant_matmul_v4_at2_at3](examples/test_aclnn_quant_matmul_v4_at2_at3.cpp) | Invokes [aclnnQuantMatmulV4](docs/aclnnQuantMatmulV4_en.md) with left-matrix per-token scaling. |
| aclnn API | [test_aclnn_quant_matmul_weight_nz](examples/test_aclnn_quant_matmul_weight_nz.cpp) | Invokes [aclnnQuantMatmulWeightNz](docs/aclnnQuantMatmulWeightNz_en.md) with FRACTAL_NZ weights. |
| aclnn API | [test_aclnn_quant_matmul_v5_hif8hif8](../quant_batch_matmul_v4/examples/arch35/test_aclnn_quant_matmul_v5_hif8hif8.cpp) | Invokes [aclnnQuantMatmulV5](../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5_en.md). The example demonstrates HIFLOAT8 quantization executed by QuantBatchMatmulV3. |

aclnnQuantMatmulV3 and aclnnQuantMatmulV4 are marked for future deprecation in their interface declarations. See the corresponding interface documents for migration information.
