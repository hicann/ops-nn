# QuantBatchMatmulV4

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |
| <term>Kirin X90 processor products</term> | √ |
| <term>Kirin 9030 processor products</term> | × |

## Function

- Operator function: Performs quantized matrix multiplication with input scaling, bias accumulation, and combinations of different input precisions.

- Version evolution: Compared with [QuantBatchMatmulV3](../quant_batch_matmul_v3/README_en.md), adds parameters such as x1_scale, x2_scale, y_scale, and offsets at different stages, and supports mixed-precision computation with 8-bit inputs and 4-bit weights.

- Formula conventions: x1 and x2 are the matrices interpreted according to transpose_x1 and transpose_x2, with shapes (M, K) and (K, N); y has shape (M, N). Scales are broadcast according to the quantization groups. A superscript j denotes the j-th group along K, with j from 0 to $\lceil K/g_K\rceil-1$, where $g_K$ is the K group size. UINT64 and INT64 contain encoded quantization parameters and must be decoded before use. An omitted bias or offset is treated as 0, and a scale not used in the computation is treated as 1.

- Formula:

  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
    - x1 is INT8, x2 is INT4, x1_scale is FLOAT32, x2_scale is UINT64, y_offset is FLOAT32, and y is FLOAT16 or BFLOAT16:

      $$
      y = ((x1 @ (x2*\mathrm{x2\_scale})) + \mathrm{y\_offset}) * \mathrm{x1\_scale}
      $$

    - Without x1_scale or bias:

      $$
      y = x1@x2 * \mathrm{x2\_scale} + \mathrm{x2\_offset}
      $$

    - With INT32 bias:

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} + \mathrm{x2\_offset}
      $$

    - With BFLOAT16 or FLOAT32 bias (no x2_offset in this case):

      $$
      y = x1@x2 * \mathrm{x2\_scale} + bias
      $$

    - With x1_scale and without bias:

      $$
      y = x1@x2 * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - With x1_scale and INT32 bias (no x2_offset in this case):

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - With x1_scale and BFLOAT16, FLOAT16, or FLOAT32 bias (no x2_offset in this case):

      $$
      y = x1@x2 * \mathrm{x2\_scale} * \mathrm{x1\_scale} + bias
      $$

    - x1 and x2 are INT8, x1_scale and x2_scale are FLOAT32, bias is FLOAT32, and y is FLOAT16 or BFLOAT16 (per-token–per-group quantization):

      $$
      y = \sum_j (x1^{(j)} @ x2^{(j)}) * \mathrm{x1\_scale}^{(j)} * \mathrm{x2\_scale}^{(j)} + bias
      $$

    - x1 and x2 are INT4, x1_scale and x2_scale are FLOAT32, x2_offset is FLOAT16, and y is FLOAT16 or BFLOAT16 (asymmetric per-token–per-group quantization):

      $$
      y = \sum_j (x1^{(j)} @ (x2^{(j)} - \mathrm{x2\_offset}^{(j)})) * \mathrm{x1\_scale}^{(j)} * \mathrm{x2\_scale}^{(j)}
      $$

  - <term>Ascend 950PR&950DT products</term>:

    - x1 and x2 use FLOAT8_E4M3FN, FLOAT8_E5M2, or HIFLOAT8; x1_scale is absent; x2_scale is INT64 or UINT64; x2_offset is absent; optional bias is FLOAT32; and y is FLOAT16, BFLOAT16, or FLOAT32:

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale}
      $$

    - In MX (Microscaling) [quantization](../../docs/en/context/quant_mode_introduction.md), x1 and x2 use FLOAT4_E2M1, FLOAT8_E4M3FN, or FLOAT8_E5M2; x1_scale and x2_scale are FLOAT8_E8M0; x2_offset is absent; and optional bias is FLOAT32:

      $$
      y = (x1* \mathrm{x1\_scale})@(x2* \mathrm{x2\_scale}) + bias
      $$

    - x1 and x2 use FLOAT8_E4M3FN, FLOAT8_E5M2, or HIFLOAT8; x1_scale and x2_scale are FLOAT32; x2_offset is absent; and optional bias is FLOAT32:

      $$
      y = (x1@x2 + bias) * \mathrm{x2\_scale} * \mathrm{x1\_scale}
      $$

    - In G-B and B-B [quantization](../../docs/en/context/quant_mode_introduction.md), x1 and x2 use FLOAT8_E4M3FN, FLOAT8_E5M2, or HIFLOAT8; x1_scale and x2_scale are FLOAT32; x2_offset and bias are absent. For x1 shape (a0, a1) and x2 shape (b0, b1), x1_scale has shape (ceil(a0 / 128), ceil(a1 / 128)) or (a0, ceil(a1 / 128)), and x2_scale has shape (ceil(b0 / 128), ceil(b1 / 128)):

      $$
      y_{pq} = \sum_{r=0}^{\lceil K/g_K\rceil-1} (x1_{pr}@x2_{rq}*(\mathrm{x1\_scale}_{pr}*\mathrm{x2\_scale}_{rq}))
      $$

      Here, p and q index blocks along M and N, r indexes groups along K, and $g_K=128$. Scales are broadcast along the corresponding rows and columns within each block.

    - x1 is FLOAT8_E4M3FN, x2 is FLOAT4_E2M1, x1_scale and x2_scale are FLOAT8_E8M0, x1_offset and x2_offset are absent, optional bias is BFLOAT16 or FLOAT16, and y is BFLOAT16 or FLOAT16:

      $$
      y = (x1 * \mathrm{x1\_scale})@(x2 * \mathrm{x2\_scale}) + bias
      $$

    - x1 is FLOAT8_E4M3FN, x2 is FLOAT4_E2M1, x1_scale is absent, x2_scale is BFLOAT16 or FLOAT16, x1_offset, x2_offset, and bias are absent, y_scale is UINT64, and y is BFLOAT16 or FLOAT16:

      $$
      y = (x1@(x2 * \mathrm{x2\_scale})) * \mathrm{y\_scale}
      $$

  - <term>Kirin X90 processor products</term>:

    - x1 is INT8, x2 is INT4 in FRACTAL_Z format, x1_scale is FLOAT32, x2_scale is UINT64, optional bias is FLOAT16, and y is FLOAT16. Scale and accumulate by K groups, treating omitted bias as 0:

      $$
      y = \left[\sum_j (x1^{(j)} @ x2^{(j)}) * \mathrm{x2\_scale}^{(j)}\right] * \mathrm{x1\_scale} + bias
      $$

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| x1 | Input | Left matrix. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 | ND |
| x2 | Input | Right matrix. FLOAT32 carries specifically preprocessed FLOAT4_E2M1 weights; it does not indicate ordinary FLOAT32 matrix multiplication. | INT8, INT4, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT32 | ND, FRACTAL_NZ, FRACTAL_NZ_C0_32, FRACTAL_NZ_C0_4, FRACTAL_Z |
| bias | Optional input | Matrix-product accumulation bias. Omitted bias adds nothing. Its accumulation position depends on the quantization mode. | INT32, BFLOAT16, FLOAT16, FLOAT32 | ND |
| x1_scale | Optional input | Left-matrix quantization scale. Required for per-token and grouped quantization. May be omitted when no separate left-matrix scale is used. | BFLOAT16, FLOAT16, FLOAT32, FLOAT8_E8M0 | ND |
| x2_scale | Optional input | Quantization scale factor for the right matrix. Required in all currently supported quantization scenarios. | BFLOAT16, FLOAT16, FLOAT32, UINT64, INT64, FLOAT8_E8M0 | ND |
| y_scale | Optional input | Quantization scale factor applied after matrix multiplication. Required when output scaling is needed; omit it otherwise. | UINT64 | ND |
| x1_offset | Optional input | Reserved quantization offset parameter for the left matrix. Currently unsupported. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| x2_offset | Optional input | Quantization offset for the right matrix. Provide the corresponding offset for asymmetric quantization; omit it otherwise. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| y_offset | Optional input | Offset correction applied after matrix multiplication. Provide it for the corresponding quantization scenario; omit it otherwise. | BFLOAT16, FLOAT16, FLOAT32 | ND |
| x2_table | Optional input | Reserved lookup table parameter for weight quantization. Currently unsupported. | INT8 | ND |
| y | Output | Result of matrix multiplication and quantization. Its data type is specified by dtype. | BFLOAT16, FLOAT16, INT8, FLOAT32 | ND |
| dtype | Attribute | Data type enumeration for output y. Supported values are 0 (FLOAT32), 1 (FLOAT16), 2 (INT8), and 27 (BFLOAT16), subject to the requirements of the product and input combination. | INT64 | - |
| compute_type | Optional attribute | <ul><li>Computation type. Supports -1 (automatic selection) or 4 (mixed-precision FLOAT8 and FLOAT4 computation).</li><li>Defaults to -1.</li></ul> | INT64 | - |
| transpose_x1 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x1: false means no transpose, and true means transpose.</li><li>Defaults to false. For supported combinations, see <a href="#constraints">Constraints</a>.</li></ul> | BOOL | - |
| transpose_x2 | Optional attribute | <ul><li>Whether to transpose the last two dimensions of x2: false means no transpose, and true means transpose.</li><li>Defaults to false. For supported combinations, see <a href="#constraints">Constraints</a>.</li></ul> | BOOL | - |
| group_size | Optional attribute | <ul><li>Quantization group size or encoded group sizes along the M, N, and K dimensions. The value depends on the quantization mode. See <a href="#constraints">Constraints</a>.</li><li>Defaults to -1. For group quantization, set it as required by the scenario.</li></ul> | INT64 | - |

## Constraints

- Empty tensors are not supported.
- group_size must match the quantization mode. Group sizes along the M, N, and K dimensions are encoded as `group_size_k | (group_size_n << 16) | (group_size_m << 32)`. G-B and B-B modes use the encodings of `[1, 128, 128]` and `[128, 128, 128]`, respectively.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>: for INT8 inputs and INT4 weights, group_size is 0 for per-channel quantization and 256 for group quantization.
- <term>Ascend 950PR&950DT products</term>: for FLOAT8 inputs and FLOAT4 weights in ND format, group_size is 32.
- The input and output support the following data type combinations:

  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y_offset | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT4 | FLOAT32 | UINT64 | Omitted | Omitted | Omitted | FLOAT32 | FLOAT16, BFLOAT16 |
    | INT8 | INT8 | Omitted | UINT64, INT64 | Omitted | Omitted | Omitted or INT32 | Omitted | FLOAT16 |
    | INT8 | INT8 | Omitted | UINT64, INT64 | Omitted or FLOAT32 | Omitted | Omitted or INT32 | Omitted | INT8 |
    | INT8 | INT8 | Omitted or FLOAT32 | BFLOAT16 | Omitted | Omitted | Omitted or INT32, BFLOAT16, FLOAT32 | Omitted | BFLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | Omitted | Omitted | Omitted or INT32, FLOAT16, FLOAT32 | Omitted | FLOAT16 |
    | INT8 | INT8 | Omitted | FLOAT32, BFLOAT16 | Omitted | Omitted | Omitted or INT32 | Omitted | FLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | Omitted | Omitted | FLOAT32 | Omitted | BFLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32, BFLOAT16 | Omitted | Omitted | Omitted or INT32, BFLOAT16, FLOAT32 | Omitted | BFLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32 | Omitted | Omitted | Omitted or INT32, FLOAT16, FLOAT32 | Omitted | FLOAT16 |
    | INT4 | INT4 | FLOAT32 | FLOAT32 | FLOAT16 | Omitted | Omitted | Omitted | BFLOAT16 |

  - <term>Ascend 950PR&950DT products</term>:

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT8 | Omitted | UINT64, INT64 | Omitted | Omitted | Omitted or INT32 | FLOAT16, BFLOAT16 |
    | INT8 | INT8 | Omitted | UINT64, INT64 | Omitted or FLOAT32 | Omitted | Omitted or INT32 | INT8 |
    | INT8 | INT8 | Omitted or FLOAT32 | FLOAT32, BFLOAT16 | Omitted | Omitted | Omitted or INT32, FLOAT32, BFLOAT16 | BFLOAT16 |
    | INT8 | INT8 | FLOAT32 | FLOAT32 | Omitted | Omitted | Omitted or INT32, FLOAT32, FLOAT16 | FLOAT16 |
    | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E4M3FN, FLOAT8_E5M2 | Omitted | UINT64, INT64 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | HIFLOAT8 | HIFLOAT8 | Omitted | UINT64, INT64 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT32 | FLOAT32 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E4M3FN, FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0 | Omitted | Omitted | Omitted or FLOAT32 | FLOAT16, BFLOAT16, FLOAT32 |
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | Omitted | Omitted | Omitted or BFLOAT16, FLOAT16 | BFLOAT16, FLOAT16 |
    | FLOAT8_E4M3FN | FLOAT4_E2M1 | Omitted | BFLOAT16, FLOAT16 | Omitted | UINT64 | Omitted | BFLOAT16, FLOAT16 |

  - <term>Kirin X90 processor products</term>:

    | x1 | x2 | x1_scale | x2_scale | x2_offset | y_scale | bias | y_offset | y |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | INT8 | INT4 | FLOAT32 | UINT64 | Omitted | Omitted | Omitted or FLOAT16 | Omitted | FLOAT16 |

    Additional constraints:

    - x2 must use FRACTAL_Z format with a storage shape of `[K/64, N/16, 16, 8]`.
    - transpose_x1 and transpose_x2 support only false.
    - group_size supports only 128 or 256.
    - Alignment requirements: `K % 64 == 0`, `N % 16 == 0`, and `K % group_size == 0`.
    - The shape of x2_scale must be `[ceil(K/group_size), N]`, where ceil rounds up to the nearest integer.
    - The number of groups, `GT=ceil(K/group_size)`, must not exceed 960.

## Invocation Description

- <term>Atlas A3 products</term>, <term>Atlas A2 products</term>:

  | Invocation method | Sample code | Description |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_quant_matmul_v5_at2_at3](examples/test_aclnn_quant_matmul_v5_at2_at3.cpp) | Calls [aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5_en.md) to demonstrate quantized computation with INT8 inputs and 4-bit weights. |

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method | Sample code | Description |
  | --- | --- | --- |
  | aclnn API | [test_aclnn_quant_matmul_v5_f8f4_nd](examples/arch35/test_aclnn_quant_matmul_v5_f8f4_nd.cpp) | Calls [aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5_en.md) to demonstrate mixed-precision FLOAT8 and FLOAT4 computation. |
  | aclnn API | [test_aclnn_quant_matmul_weight_nz_f8mxf4](../quant_batch_matmul_v3/examples/arch35/test_aclnn_quant_matmul_weight_nz_f8mxf4.cpp) | Calls [aclnnQuantMatmulWeightNz](../quant_batch_matmul_v3/docs/aclnnQuantMatmulWeightNz_en.md) to demonstrate mixed-precision FLOAT8 and FLOAT4 computation with FRACTAL_NZ weights. |
  | GE graph mode | [test_geir_quant_batch_matmul_v4](examples/arch35/test_geir_quant_batch_matmul_v4.cpp) | Builds and invokes a graph using the [QuantBatchMatmulV4 operator IR](op_graph/quant_batch_matmul_v4_proto.h). |

aclnnQuantMatmulV5 calls QuantBatchMatmulV3 or QuantBatchMatmulV4 depending on the input combination. For interface parameters and layout requirements, see [aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5_en.md).
