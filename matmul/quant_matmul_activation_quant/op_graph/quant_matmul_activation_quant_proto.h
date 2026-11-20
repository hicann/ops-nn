/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_activation_quant_proto.h
 * \brief
 */
#ifndef OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_
#define OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
// clang-format off
/**
* @brief The fusion operator of QuantBatchMatmul, activation and mxQuant.

* @par Inputs:
 * @li x1: A matrix tensor. Must be one of the following types: float8_e5m2, float8_e4m3fn, float4_e2m1 \n
           when the data type is float8_e5m2, float8_e4m3fn, float4_e2m1, the format only supports ND format. \n
           - In ND format, the shape ranges from 2D to 6D. When transpose_x1 is false, \n
            the shape is (batch,m,k); when true, it is (batch,k,m), where batch is optional. \n
 * @li x2: A matrix tensor. Must be one of the following types: float8_e4m3fn, float8_e5m2, float4_e2m1. \n
           when the data type is float8_e4m3fn or float4_e2m1, the format supports NZ and ND format. \n
           when the data type is float8_e5m2, the format only supports ND format. \n
          - In ND format, the shape ranges from 2D to 6D. When transpose_x2 is false, \n
            the shape is (batch,k,n), where batch is optional. \n
          - In NZ (Ascend affinity) format, the shape ranges from 4D to 8D. \n
              - When transpose_x2 is true, the shape is (batch,x2_k1,x2_n1,x2_n0,x2_k0), where batch is optional, x2_k0
= 32 for float8_e4m3fn or 64 for float4_e2m1, and x2_n0 = 16 (in logical elements). \n
              - When transpose_x2 is false, the shape is (batch,x2_n1,x2_k1,x2_k0,x2_n0), where batch is optional, x2_k0
= 16, and x2_n0 = 32 for float8_e4m3fn or 64 for float4_e2m1 (in logical elements). \n
              - When x1 is ND format, k in the shape of x1 and the shape of x2 must meet the following requirement: \n
                    ceil(k / x2_k0) == x2_k1. \n
* @li bias: An optional matrix tensor. Must be float32, supports ND format.
            The shape is 1D (t,) or 3D (batch, 1, n),
            with t equal to n, where n is the same as that of x2. The 3D bias form requires a 3D output.
            For SwiGLU,
            bias is applied to the full pre-activation width before splitting.
* @li x1_scale: Required in MX mode. Uses float8_e8m0 and ND format, with rank 3 to 7.
* Its batch dimensions must match x1. The last three dimensions are (m, z, 2) when transpose_x1 is false,
* or (z, m, 2) when true, where z = ceil(k / 64). \n
* @li x2_scale: Required in MX mode. Uses float8_e8m0 and ND format, with rank 3 to 7.
* Its batch dimensions must match x2. The last three dimensions are (z, n, 2) when transpose_x2 is false,
* or (n, z, 2) when true, where z = ceil(k / 64). \n
* @li x3: Optional BF16 ND multiplier, required when activation_type="sigmoid".
* When x3 is provided, x1, x2, x3 and y must have 2D logical shapes, and x3 must match y.
* Batch dimensions (including size-1 dimensions) and broadcasting are not supported.
* Sigmoid and multiplication use FP32; the product is rounded to BF16 before MX quantization.
* For other activations, x3 must be absent.
* @par Attributes:
* @li transpose_x1: A bool. If true, x1 is stored as [..., k, m] and transposed to [..., m, k] for multiplication.
* Default: false. For SwiGLU and SiLU, true is supported with ND x2; WeightNZ x2 requires false.
* @li transpose_x2: A bool. If true, x2 is stored as [..., n, k] and transposed to [..., k, n] for multiplication.
* Default: false.
* @li group_size: An optional Int encoding the input quantization group sizes along M, N and K. \n
* The group_size is composed of the group_size_m, group_size_n, and group_size_k, total occupying 48 bits.
* 0-15 bits of group_size indicate group_size_k, 16-31 bits indicate group_size_n, 32-47 bits indicate group_size_m,
* 48-63 bits of group_size must be zero. \n
* A group_size of 0 selects [group_size_m, group_size_n, group_size_k] = [1, 1, 32].
* Otherwise, only the encoding 4295032864 for [1, 1, 32] is supported. \n
* Final group_size_m, group_size_n, group_size_k should satisify following requirements: \n
* In mx quantification, the supported final [group_size_m, group_size_n, group_size_k] combinations are [1, 1, 32]. \n
* @li activation_type: An optional string. The activation algorithm to use: 'gelu_tanh', 'gelu_erf', 'swiglu',
'silu' or 'sigmoid', default is 'gelu_tanh'.
* @li y_dtype: An optional Int. The output dtype must match x1. Defaults to DT_FLOAT8_E4M3FN.
* @li quant_mode: An optional string. Declare the quant mode.Support mx. Defaults to "mx".
* @li round_mode: An optional string. Defaults to "rint".
* @li scale_alg: An optional int.The algorithm for the scale in quantization.Default to 0.
* Value 0 uses OCP Microscaling for MXFP8/MXFP4; value 1 uses the cuBLAS scale algorithm for MXFP8;
* value 2 uses the dynamic dtype range (DYN_DTYPE_RANGE) algorithm for MXFP4.
* @li dst_type_max: An optional Float used only when scale_alg is 2. Supports 0 or [6, 12]; 0 is equivalent to 6.
* Defaults to 0.


* @par Outputs:
 * @li y: A matrix tensor. The data type is float8_e5m2, float8_e4m3fn, float4_e2m1. The format supports ND.
 * The shape is (batch, m, n) for GELU and SiLU, or (batch, m, n / 2) for SwiGLU. \n
* @li y_scale: An ND tensor of type FLOAT8_E8M0, with shape (batch..., m, ceil(h / 64), 2).
* Here h = n for GELU/SiLU and h = n / 2 for SwiGLU; batch dimensions match the broadcast output y.
* Each scale represents 32 output elements, and two adjacent scales share the final dimension of size 2.
* An unused scale slot is filled with byte encoding 0 (E8M0 value 2^-127).


* @attention Constraints:
* @li The shape of bias should be 1D when the shape of out is 2D, 4D, 5D or 6D, and the shape of bias should be 1D or 3D
* when the out shape is 3D.
* @li Inputs does not support tensor with dimension size 0.
* @li When x2 is ND format, x1 should be ND format.
* @li When activation_type is 'swiglu', 'silu' or 'sigmoid', x1 and x2 must be FP8, x1 must be ND,
* x2 must be ND or FLOAT8_E4M3FN NZ; transpose_x1 is supported for ND x2 and must be false for WeightNZ x2;
* y_dtype must match x1; round_mode must be 'rint'; scale_alg must be 0 or 1.
* For SwiGLU, the pre-activation N must be divisible by 64.
* @li In mx quantification, when x1 type and x2 type are both float8_e4m3fn/float8_e5m2 and x1_scale/x2_scale types are
float8_e8m0:
 *      - supported final [group_size_m, group_size_n, group_size_k] is [1, 1, 32].
 * @li In mx quantification, when x1 type and x2 type are both float4_e2m1 and x1_scale/x2_scale types are
 * float8_e8m0 (MXFP4 scenario):
 *      - x1 and x2 inner axis (the last dimension of view shape, independent of transpose_x1/transpose_x2) must be
even.
 *      - k must be greater than 2.
 *      - y's N dimension must be even.
 *      - supported final [group_size_m, group_size_n, group_size_k] is [1, 1, 32].
 *      - scale_alg only supports 0 and 2. When scale_alg is 2, dst_type_max supports 0.0 and 6.0-12.0.
 *      - round_mode supports "rint", "floor", "round".
 *      - When x2 is NZ format, x1 does not support transpose, i.e. transpose_x1 cannot be true.
 * @li Only weight supports ND and NZ format on Ascend 950PR&950DT products. All other inputs and outputs only support
ND
format.
* @li The following are the supported data type combinations by platform.

* - Ascend 950PR&950DT products:
*
* | x1            | x2            | bias         | x1_scale    | x2_scale    | out           | y_scale     |
* | :-----------: | :-----------: | :----------: | :---------: | :---------: | :-----------: | :---------: |
* | float8_e4m3fn | float8_e4m3fn | null/float32 | float8_e8m0 | float8_e8m0 | float8_e4m3fn | float8_e8m0 |
* | float8_e5m2   | float8_e5m2   | null/float32 | float8_e8m0 | float8_e8m0 | float8_e5m2   | float8_e8m0 |
* | float8_e4m3fn | float8_e5m2   | null/float32 | float8_e8m0 | float8_e8m0 | float8_e4m3fn | float8_e8m0 |
* | float8_e5m2   | float8_e4m3fn | null/float32 | float8_e8m0 | float8_e8m0 | float8_e5m2   | float8_e8m0 |
* | float4_e2m1   | float4_e2m1   | null/float32 | float8_e8m0 | float8_e8m0 | float4_e2m1   | float8_e8m0 |
*
* - Logical input shapes in MX mode, where z = ceil(k / 64):
*
* | Input | Transpose false | Transpose true |
* | ----- | --------------- | -------------- |
* | x1 | (batch1..., m, k) | (batch1..., k, m) |
* | x1_scale | (batch1..., m, z, 2) | (batch1..., z, m, 2) |
* | x2 | (batch2..., k, n) | (batch2..., n, k) |
* | x2_scale | (batch2..., z, n, 2) | (batch2..., n, z, 2) |
*
* batch1 and batch2 broadcast from the right. Input scales retain their corresponding input batch dimensions.
* The FP4 and WeightNZ transpose restrictions above still apply.
*
*/
// clang-format on
REG_OP(QuantMatmulActivationQuant)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT32}))
    // Keep optional registration for interface compatibility; both scales are mandatory in MX mode.
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(x3, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .ATTR(activation_type, String, "gelu_tanh")
    .ATTR(y_dtype, Int, DT_FLOAT8_E4M3FN)
    .ATTR(quant_mode, String, "mx")
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0)
    .OP_END_FACTORY_REG(QuantMatmulActivationQuant)
} // namespace ge

#endif // OPS_QUANT_MATMUL_ACTIVATION_QUANT_PROTO_H_
