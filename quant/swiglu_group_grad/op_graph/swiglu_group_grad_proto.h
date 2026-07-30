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
 * \file swiglu_group_grad_proto.h
 * \brief SwigluGroupGrad operator proto definition
 *
 * IR definition:
 *   Required inputs:  grad_y (T,H)/(B,S,H), x (T,2H)/(B,S,2H)
 *   Optional inputs:  weight (T,1)/(B,S,1) FP32, y_origin (T,H)/(B,S,H), group_index (G,) INT64
 *   Required output:  grad_x (T,2H)/(B,S,2H)
 *   Optional output:  grad_weight (T,1)/(B,S,1) FP32
 *   Attribute:        clamp_limit (Float, default=0 → no clamp)
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Compute the SwigluGroupGrad — gradient of ClampedSwiglu activation.

* @par Inputs:
* five inputs, including:
* @li grad_y: A required 2D or 3D Tensor of shape (..., H). Must be one of: bfloat16, float16, float32.
* @li x: A required Tensor of shape (..., 2H) and the same rank as grad_y.
 *     Must be one of: bfloat16, float16, float32.
 * @li weight: An optional Tensor of shape (..., 1) with dtype float32. MoE top-k routing weight.
 *     It must be provided together with y_origin.
 * @li y_origin: An optional Tensor of shape (..., H) with same dtype as grad_y. Forward output y,
 *     including weight multiplication when weight is present.
 *     It must be provided together with weight.
* @li group_index: An optional non-empty Tensor of shape (G,), G > 0, with dtype int64. Token/batch count per group.

* @par Outputs:
* two outputs, including:
* @li grad_x: A required Tensor of shape (..., 2H) with same dtype as grad_y. Gradient of x.
* @li grad_weight: An optional Tensor of shape (..., 1) with dtype float32. Gradient of weight.

* @par Attributes:
* one attribute:
* @li clamp_limit: An optional Float. Clipping threshold c; default 0 means no clamp (c=+∞).
*/
REG_OP(SwigluGroupGrad)
    .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(y_origin, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad_weight, TensorType({DT_FLOAT}))
    .ATTR(clamp_limit, Float, 0)
    .OP_END_FACTORY_REG(SwigluGroupGrad)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_GRAD_PROTO_H_
