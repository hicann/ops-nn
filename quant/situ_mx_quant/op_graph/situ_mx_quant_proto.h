/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_mx_quant_proto.h
 * \brief Situ activation combined with dynamic MX quantization operator prototype
 */

#ifndef SITU_MX_QUANT_PROTO_H_
#define SITU_MX_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Performs Situ activation followed by dynamic MX quantization on input tensor.
 * This fused operator first computes Situ activation by splitting input along the last dimension,
 * then applies block-wise MX quantization along the specified axis.
 *
 * @par Inputs:
 * @li x: An input tensor of type float16 or bfloat16.
 * The size of the last dimension must be divisible by 2.
 * Supports 1-7 dimensional tensors.
 *
 * @par Attributes:
 * @li beta: An optional float. Beta parameter for Situ activation. Must be greater than 0. Defaults to 1.0.
 * @li linear_beta: An optional float. Linear beta parameter for Situ activation.
 * When <= 0, the linear_beta transformation is not applied. Defaults to 0.0.
 * @li activate_left: An optional bool. When true, gate is the first half and up is the second half.
 * When false, gate is the second half and up is the first half. Defaults to false.
 * @li axis: An optional int. Axis along which to perform block-wise quantization.
 * Currently only supports -1 (last axis). Defaults to -1.
 * @li dst_type: An optional int. Target quantization data type.
 * 40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2. Defaults to 40 (FP4_E2M1).
 * @li round_mode: An optional string. Rounding mode for quantization.
 * Supports "rint", "round", "floor". FP8 output only supports "rint". Defaults to "rint".
 *
 * @par Outputs:
 * @li y: Quantized output tensor after Situ activation.
 * Shape is same as input except last dimension is halved.
 * Data type is float4_e2m1, float4_e1m2, float8_e4m3fn, or float8_e5m2.
 * @li y_scale: Scale factors for each quantization block. Data type is float8_e8m0.
 * Shape: y_shape with axis dimension replaced by ceil(y_shape[axis] / 64), plus trailing dim of 2.
 *
 * @par Constraints:
 * @li Input last dimension must be divisible by 2.
 * @li axis must be -1 (last axis).
 * @li beta must be greater than 0.
 * @li dst_type must be 40 (FP4_E2M1), 41 (FP4_E1M2), 36 (FP8_E4M3FN), or 35 (FP8_E5M2).
 * @li When dst_type is FP8, round_mode must be "rint".
 * @li When dst_type is FP4, output last dim must be even.
 *
 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
 */
REG_OP(SituMxQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT8_E8M0}))
    .ATTR(beta, Float, 1.0f)
    .ATTR(linear_beta, Float, 0.0f)
    .ATTR(activate_left, Bool, false)
    .ATTR(axis, Int, -1)
    .ATTR(dst_type, Int, DT_FLOAT4_E2M1)
    .ATTR(round_mode, String, "rint")
    .OP_END_FACTORY_REG(SituMxQuant)

} // namespace ge

#endif // SITU_MX_QUANT_PROTO_H_
