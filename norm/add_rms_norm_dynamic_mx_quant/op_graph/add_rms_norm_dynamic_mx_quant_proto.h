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
 * \file add_rms_norm_dynamic_mx_quant_proto.h
 * \brief
 */
#ifndef NORM_ADD_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_
#define NORM_ADD_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Fused Operator of AddRmsNorm and DynamicMxQuant.
* Computes x = x1 + x2, applies RMS normalization with gamma/beta, then quantizes.
* Supports various 4-bit and 8-bit floating point output formats.
* Calculating process: \n
*  x = x1 + x2 \n
*  rstd = np.rsqrt(np.mean(np.power(x, 2), reduce_axis, keepdims=True) + epsilon)) \n
*  rmsnorm_out = x * rstd * gamma \n
*  the normalized result using block-wise MX scaling factors along the last axis.


* @par Inputs:
* @li x1: A tensor for add compute. Support dtype: float16, bfloat16, support format: ND.
* The shape supports at least 1 dimension, and at most 7 dimensions.
* @li x2: A tensor for add compute. Support dtype: float16, bfloat16, support format: ND.
* The shape must be the same as x1.
* @li gamma: A tensor for rms norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* The shape must match the normalized dimension of x1.
* @li beta: An optional tensor for rms norm weight params. Support dtype: float32, float16, bfloat16, support format: ND.
* The shape must match the normalized dimension of x1.

* @par Attributes:
* @li epsilon: An optional attribute for numerical stability in rms norm, the type is float32. Defaults to 1e-6.
* @li scale_alg: An optional int.The algorithm for the scale in quantization. Default to 0.
* Support (OCP , count 0) or (nvidia-cuBLAS , count 1).
* @li round_mode: An optional string. Defaults to "rint".
* @li dst_type: An optional attribute. Declare the output y dtype. Support FLOAT4_E2M1, FLOAT4_E1M2,
* FLOAT8_E4M3FN or FLOAT8_E5M2. Defaults to FLOAT4_E2M1 (40).
* @li output_rstd: An optional attribute. Defaults to "false". Whether to output Rstd.

* @par Outputs:
* @li y: Quantize result.
*     A tensor. Support dtype: Support FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN or FLOAT8_E5M2, support format: ND.
* @li x: Describing the result of x1 + x2.
*     A tensor. Support dtype: float16, bfloat16, support format: ND.
* @li mxscale: An output tensor of type FLOAT8_E8M0. Shape needs to meet the following conditions: \n
* - rank(mxscale) = rank(x) + 1.
* - axis_change = axis if axis >= 0 else axis + rank(x).
* - mxscale.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2.
* - mxscale.shape[rank(x)] = 2.
* - Other dimensions match input x.
* @li rstd: A tensor. Describing the reciprocal of (x1 + x2)'s standard deviation.
*           Support dtype: float32, support format: ND.

* @attention Constraints:
* @li When dst_type is DT_FLOAT8_E5M2 or DT_FLOAT8_E4M3FN, round_mode only supports "rint".
* @li When dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, round_mode supports "rint", "floor" and "round".
* @li If dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, the input x last dimension of the shape must be divisible by 2.
* @li If dst_type is DT_FLOAT4_E2M1 or DT_FLOAT4_E1M2, the scale_alg only support (OCP , count 0).
*/
REG_OP(AddRmsNormDynamicMxQuant)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(gamma, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .OUTPUT(rstd, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 1e-6)
    .ATTR(scale_alg, Int, 0)
    .ATTR(round_mode, String, "rint")
    .ATTR(dst_type, Int, 40)
    .ATTR(output_rstd, Bool, false)
    .OP_END_FACTORY_REG(AddRmsNormDynamicMxQuant)

} // namespace ge

#endif // NORM_ADD_RMS_NORM_DYNAMIC_MX_QUANT_PROTO_H_
