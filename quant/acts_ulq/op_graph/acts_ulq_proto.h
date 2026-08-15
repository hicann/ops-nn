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
 * \file acts_ulq_proto.h
 * \brief ActsULQ GE IR 算子注册
 */
#ifndef OPS_OP_PROTO_INC_ACTSULQ_H_
#define OPS_OP_PROTO_INC_ACTSULQ_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
*@brief Activations Universal Linear Quantization. \n

*@par Inputs:
*@li x: A Tensor of feature map. Must be one of the following types: float16, float32.
* Supports empty Tensor. The format support ND. Supports 0-8 dimensions.
*@li clamp_min: A Tensor of the min clamp value of feature map. Must be one of the
* following types: float16, float32, and the data type must be the same as "x".
* Does not support empty Tensor. The format support ND. The shape must be 1.
*@li clamp_max: A Tensor of the max clamp value of feature map. Must be one of the
* following types: float16, float32, and the data type must be the same as "x".
* Does not support empty Tensor. The format support ND. The shape must be 1. \n

*@par Attributes:
*@li fixed_min: (Optional) Bool, default false. Whether to fix the lower bound to zero.
* When true, ori_clip_min = 0; when false, ori_clip_min = min(clamp_min, 0).
*@li num_bits: (Optional) Int, default 8. Quantization bit-width. Only 8 is supported currently. \n

*@par Outputs:
*@li y: A Tensor of the fake quant feature map. Must be one of the following types:
* float16, float32, and the data type must be the same as "x". Supports empty Tensor.
* The format support ND. The shape must be the same as "x".
*@li clamp_min_mask: A Tensor of the lower-bound mask, value 1.0 (x >= clip_min) or 0.0.
* Must be one of the following types: bool, float16, float32. Supports empty Tensor.
* The format support ND. The shape must be the same as "y".
*@li clamp_max_mask: A Tensor of the upper-bound mask, value 1.0 (x <= clip_max) or 0.0.
* Must be one of the following types: bool, float16, float32. Supports empty Tensor.
* The format support ND. The shape must be the same as "y".
*@li x_clamped_loss: A Tensor of the clamp loss. Must be one of the following types:
* float16, float32. Supports empty Tensor. The format support ND. The shape must be
* the same as "y". \n

*@par Third-party framework compatibility
*Compatible with MindSpore operator ActsULQ.
*/
#ifndef OPS_PROTO_DEF_ACTSULQ
#define OPS_PROTO_DEF_ACTSULQ
REG_OP(ActsULQ)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(clamp_min, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(clamp_max, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(clamp_min_mask, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(clamp_max_mask, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(x_clamped_loss, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(fixed_min, Bool, false)
    .ATTR(num_bits, Int, 8)
    .OP_END_FACTORY_REG(ActsULQ)
#endif

} // namespace ge

#endif // OPS_OP_PROTO_INC_ACTSULQ_H_
