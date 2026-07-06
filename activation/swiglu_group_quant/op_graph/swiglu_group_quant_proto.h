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
 * \file swiglu_group_quant_proto.h
 * \brief SwiGLU activation followed by grouped low-bit quantization.
 */

#ifndef QUANT_SWIGLU_GROUP_QUANT_PROTO_H_
#define QUANT_SWIGLU_GROUP_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
 * @brief Performs SwiGLU activation followed by Block FP8, MX FP8, MX FP4, or HiFloat8 quantization.
 *
 * @par Inputs:
 * @li x: Required tensor of type float16 or bfloat16. quant_mode=3 also supports float32. The last dimension
 * is split into two equal parts for SwiGLU and must be divisible by 256.
 * @li weight: Optional float32 tensor. Per-token weight multiplied into the SwiGLU result before quantization.
 * @li group_index: Optional int64 tensor. Count-mode group token numbers.
 * @li scale: Optional float32 tensor. Reserved for static quantization modes.
 *
 * @par Attributes:
 * @li dst_type: Optional int. Target quantized dtype. Supports FLOAT8_E4M3FN, FLOAT8_E5M2,
 * FLOAT4_E2M1, FLOAT4_E1M2 and HIFLOAT8. Defaults to FLOAT8_E4M3FN.
 * @li quant_mode: Optional int. 0 means Block FP8 quantization, 1 means MX quantization,
 * 3 means HiFloat8 dynamic quantization. Defaults to 0.
 * @li block_size: Optional int. 0 selects the mode default. Supports 128 for Block FP8 and 32 for MX.
 * Defaults to 0.
 * @li round_scale: Optional bool. MX quantization requires true. Defaults to false.
 * @li clamp_limit: Optional float. Defaults to -1.0 for quant_mode 0/1 and 0.0 for quant_mode 3,
 * both of which disable clamp. If set to a positive value, clamps SwiGLU inputs before activation.
 * @li dst_type_max: Optional float. Maximum finite value used by quant_mode=3 scale calculation.
 * @li output_origin: Optional bool. Writes the pre-quantized SwiGLU result to y_origin when supported.
 * Defaults to false.
 *
 * @par Outputs:
 * @li y: Quantized output tensor. FP8 output shape is input shape with the last dimension halved.
 * FP4 output packs two values in one byte, so its last dimension is input_last_dim / 4.
 * @li y_scale: Scale tensor. float32 for Block FP8 and HiFloat8 dynamic quantization, float8_e8m0 for MX.
 * @li y_origin: SwiGLU result before quantization, with the same dtype as x and last dimension halved.
 *
 * @par Third-party framework compatibility
 * It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
 */
REG_OP(SwigluGroupQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2, DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_HIFLOAT8}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y_origin, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(dst_type, Int, DT_FLOAT8_E4M3FN)
    .ATTR(quant_mode, Int, 0)
    .ATTR(block_size, Int, 0)
    .ATTR(round_scale, Bool, false)
    .ATTR(clamp_limit, Float, -1.0f)
    .ATTR(dst_type_max, Float, 15.0f)
    .ATTR(output_origin, Bool, false)
    .OP_END_FACTORY_REG(SwigluGroupQuant)

} // namespace ge

#endif // QUANT_SWIGLU_GROUP_QUANT_PROTO_H_