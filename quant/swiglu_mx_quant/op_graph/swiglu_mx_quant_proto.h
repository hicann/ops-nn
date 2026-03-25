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
 * \file swiglu_mx_quant_proto.h
 * \brief SwiGLU activation function combined with dynamic block-wise MX quantization
 */

#ifndef SWIGLU_MX_QUANT_PROTO_H_
#define SWIGLU_MX_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs SwiGLU activation followed by dynamic MX quantization on input tensor.
* This fused operator first computes SwiGLU activation by splitting input along activate_dim,
* then applies block-wise quantization along the specified axis.
*
* @par Inputs:
* @li x: An input tensor. Must be one of the following types: float16, bfloat16.
* The size of the dimension specified by activate_dim must be divisible by 2.
* Supports 2-7 dimensional tensors.
* @li group_index: An optional tensor for reserved parameter. Must be one of the following types: int32, int64.
* This parameter is reserved for future use and is not currently used in computation.

* @par Attributes:
* @li activate_left: An optional bool. Reserved parameter for SwiGLU activation side. Defaults to false.
* @li activate_dim: An optional int. Dimension along which to split input for SwiGLU.
* Must be last or second-to-last dimension. Defaults to -1.
* @li swiglu_mode: An optional int. Reserved parameter for SwiGLU variant mode. Defaults to 0. When swiglu_mode = 1, clamp_limit must greater than 0
* @li clamp_limit: An optional float. Reserved parameter for clamp limit in SwiGLU variant. Defaults to 7.0.
* @li glu_alpha: An optional float. Reserved parameter for alpha value in SwiGLU variant. Defaults to 1.702.
* @li glu_bias: An optional float. Reserved parameter for bias value in SwiGLU variant. Defaults to 1.0.
* @li group_mode: An optional int. Group index mode. Effective when group_index is provided.
* 0=count mode, 1=cumsum mode. Defaults to 0.Currently only supports 0.
* @li axis: An optional int. Axis along which to perform block-wise quantization.
* Must be last or second-to-last dimension. Defaults to -1.
* @li dst_type: An optional int. Target quantization data type.
* 40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2. Defaults to 40 (FP4_E2M1).
* @li round_mode: An optional string. Rounding mode for quantization.
* Supports "rint", "floor", "round". Defaults to "rint". When dst_type = 35 or 36, round_mode must be "rint".
* @li scale_alg: An optional int. Algorithm for computing scale factors.
* 0=OCP, 1=cuBLAS, 2=RNE. Defaults to 0.When dst_type = 40 or 41, scale_alg must be 0.
* @li max_dtype_value: An optional float. Reserved parameter for maximum dtype value. Used when scale_alg=2 and dst_type=FP4_E1M2. Defaults to 0.

* @par Outputs:
* @li y: Quantized output tensor after SwiGLU activation.
* Shape is same as input except activate_dim dimension is halved.
* Data type is one of: float4_e2m1, float4_e1m2, float8_e4m3fn, float8_e5m2.
* @li mxscale: Scale factors for each quantization block. Data type is float8_e8m0.
* Shape calculation: \n
* - Let act_shape be the shape after SwiGLU (input shape with activate_dim halved) \n
* - axis_idx = axis if axis >= 0 else axis + rank(act_shape) \n
* - mxscale.shape = act_shape \n
* - mxscale.shape[axis_idx] = ceil(act_shape[axis_idx] / 32) \n
* - mxscale.shape[-1] = (mxscale.shape[-1] + 1) // 2  (packed storage) \n
* - mxscale.shape = mxscale.shape + [2]  (last dimension expanded to 2 for real/imaginary parts)

* @par Constraints:
* @li Input dimension specified by activate_dim must be divisible by 2.
* @li activate_dim and axis must be last or second-to-last dimension, example -1 or -2;
* @li When dst_type is FP4 (40 or 41), the last dimension must be divisible by 4.
* @li When dst_type is FP8_E4M3FN (36) or FP8_E5M2 (35), round_mode supports "rint".
* @li When dst_type is FP4_E2M1 (40) or FP4_E1M2 (41), round_mode supports "rint", "floor", "round".

* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
*/
REG_OP(SwigluMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(mxscale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(activate_dim, Int, -1)
    .ATTR(activate_left, Bool, false)
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 7.0f)
    .ATTR(glu_alpha, Float, 1.702f)
    .ATTR(glu_bias, Float, 1.0f)
    .ATTR(group_mode, Int, 0)
    .ATTR(axis, Int, -1)
    .ATTR(dst_type, Int, DT_FLOAT4_E2M1)
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(max_dtype_value, Float, 0.0f)
    .OP_END_FACTORY_REG(SwigluMxQuant)

} // namespace ge

#endif // SWIGLU_MX_QUANT_PROTO_H_
