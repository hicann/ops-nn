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
 * \file swiglu_group_quant_grad_proto.h
 * \brief SwiGLU Group Dynamic Quant Backward operator prototype
 */
#ifndef OPS_QUANT_SWIGLU_GROUP_QUANT_GRAD_PROTO_H_
#define OPS_QUANT_SWIGLU_GROUP_QUANT_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief SwiGLU Group Dynamic Quant Backward operator.

* @par Inputs:
* @li grad_y: Gradient input tensor. Must be one of the following types: float32,float16,bfloat16, has format ND.
* @li x: Forward pass input tensor. Must be one of the following types: float32,float16,bfloat16, has format ND.
* @li weight: Optional tensor. topk weight tensor. Type is float32, has format ND.
* @li y_origin: Optional tensor. Forward pass output before quantization.
        Must be one of the following types: float32,float16,bfloat16, has format ND.
* @li group_index: Optional tensor. Group index tensor for dynamic quantization. Type is int64, has format ND.

* @par Attributes:
* @li clamp_limit: Optional float. Clamp value for gradient mask, default is -1.0 (no clamp).
*     If set to a positive value, clamps SwiGLU inputs before activation. Must be -1.0 or > 0.

* @par Outputs:
* @li grad_x: Gradient of x tensor. Same type as input x, has format ND.
* @li grad_weight: Optional output. Gradient of weight tensor. Type is float32, has format ND.
*/
REG_OP(SwigluGroupQuantGrad)
    .INPUT(grad_y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(y_origin, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(grad_weight, TensorType({DT_FLOAT}))
    .ATTR(clamp_limit, Float, -1.0f)
    .OP_END_FACTORY_REG(SwigluGroupQuantGrad)
} // namespace ge

#endif // OPS_QUANT_SWIGLU_GROUP_QUANT_GRAD_PROTO_H_