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
 * \file swiglu_group_proto.h
 * \brief SwiGLU activation with optional per-token weight and grouped tokens.
 */

#ifndef ACTIVATION_SWIGLU_GROUP_PROTO_H_
#define ACTIVATION_SWIGLU_GROUP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Performs SwiGLU activation.
*
* @par Inputs:
* @li x: Required tensor of type float16, bfloat16 or float32. The last dimension is split into two
* equal parts for SwiGLU and must be divisible by 2.
* @li weight: Optional float32 tensor. Per-token weight multiplied into the SwiGLU result.
* @li group_index: Optional int64 tensor. Count-mode group token numbers.
*
* @par Attributes:
* @li clamp_limit: Optional float. Defaults to -1.0, which disables clamp. If set to a positive value,
* clamps SwiGLU inputs before activation.
*
* @par Outputs:
* @li y: SwiGLU result tensor with the same dtype as x and last dimension halved.
*
* @par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe, ONNX, TensorFlow, or PyTorch.
*/
REG_OP(SwigluGroup)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(clamp_limit, Float, -1.0f)
    .OP_END_FACTORY_REG(SwigluGroup)

} // namespace ge

#endif // ACTIVATION_SWIGLU_GROUP_PROTO_H_
