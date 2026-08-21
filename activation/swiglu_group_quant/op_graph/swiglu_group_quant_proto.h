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
 * @li x: Required tensor. float16 or bfloat16 for quant_mode 0/1; float16, bfloat16 or float32 for
 * quant_mode 2/3. The rank must be in [2, 8] ([2, 7] for quant_mode 1), empty tensors are not
 * supported, and the last dimension is split into two equal parts for SwiGLU and must be greater than
 * or equal to 256 and divisible by 256.
 * @li weight: Optional float32 tensor. Per-token weight multiplied into the SwiGLU result before
 * quantization. The rank must be in [1, 8], empty tensors are not supported, and the element count must
 * equal the product of all x dims except the last one.
 * @li group_index: Optional int64 tensor. Count-mode group token numbers. It must be 1D, its element
 * values must be greater than or equal to 0, and empty tensors are not supported.
 * @li scale: Optional float32 tensor. Static quantization input (invScale) used by quant_mode 2. Its
 * shape must be [G] when group_index is present and [1] otherwise, and empty tensors are not supported.
 *
 * @par Attributes:
 * @li dst_type: Optional int. Target quantized dtype. It is only effective for quant_mode 0/1, and
 * supports 35 (FLOAT8_E5M2), 36 (FLOAT8_E4M3FN), 40 (FLOAT4_E2M1) and 41 (FLOAT4_E1M2). quant_mode 1
 * is required when dst_type is 40 or 41. quant_mode 2/3 always quantize to HIFLOAT8 and ignore this
 * attribute. Defaults to FLOAT8_E4M3FN.
 * @li quant_mode: Optional int. 0 means Block FP8 quantization, 1 means MX quantization, 2 means
 * HiFloat8 static quantization, 3 means HiFloat8 dynamic quantization. Defaults to 0.
 * @li block_size: Optional int. 0 selects the mode default. Supports 128 for Block FP8 and 32 for MX.
 * Defaults to 0.
 * @li round_scale: Optional bool. MX quantization requires true. Defaults to false.
 * @li clamp_limit: Optional float. Defaults to -1.0, which disables clamp. If set to a positive value,
 * clamps SwiGLU inputs before activation.
 * @li dst_type_max: Optional float. Maximum finite value used by quant_mode=3 scale calculation.
 * Defaults to 15.0.
 * @li output_origin: Optional bool. Writes the pre-quantized SwiGLU result to y_origin. quant_mode 0/1
 * only support false, quant_mode 2/3 support both true and false. Defaults to false.
 *
 * @par Outputs:
 * @li y: Quantized output tensor. The shape is the input x shape with the last dimension halved for all
 * quant modes. FP4 physical storage packs two values in one byte via its dtype, so it occupies D/4 bytes.
 * @li y_scale: Scale tensor. float32 for Block FP8, HiFloat8 static and HiFloat8 dynamic quantization,
 * float8_e8m0 for MX.
 * @li y_origin: SwiGLU result before quantization, with the same dtype and rank as x, the same dims as x
 * except that the last dimension is halved.
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
