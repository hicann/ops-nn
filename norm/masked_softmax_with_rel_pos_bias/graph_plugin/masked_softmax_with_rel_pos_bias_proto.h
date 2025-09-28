/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_rms_norm_rope_cache_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_
#define OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief swin_transformer model specific structure.Operator only supports swin_transformer.

* @par Inputs:
* Three inputs, including:
* @li x: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be (B*W, N, S1, S2) or (B, W, N, S1, S2).
* @li atten_mask: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
                  the shape should be (W, S1, S2) or (W, 1, S1, S2) or (1, W, 1, S1, S2)
* @li relative_pos_bias: An ND Tensor. Must be one of the following types: float16, float, bfloat16.
                         the shape sholud be (N, S1, S2) or (1, N, S1, S2) or (1, 1, N, S1, S2)

* @par Attributes:
* @li scale_value: A optional attribute, the type is float. Defaults to 1.0.
* @li inner_precision_mode: A optional attribute, the type is int. Defaults to 0, reserved field.

* @par Outputs:
* One output, including:
* @li y: An ND Tensor. Must be one of the following types: float16, float, bfloat16,
         the shape should be same with x.
*/
REG_OP(MaskedSoftmaxWithRelPosBias)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .INPUT(relative_pos_bias, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}))
    .ATTR(scale_value, Float, 1.0)
    .ATTR(inner_precision_mode, Int, 0)
    .OP_END_FACTORY_REG(MaskedSoftmaxWithRelPosBias)
} // namespace ge

#endif // OPS_OP_PROTO_INC_KV_RMS_NORM_ROPE_CACHE_H_