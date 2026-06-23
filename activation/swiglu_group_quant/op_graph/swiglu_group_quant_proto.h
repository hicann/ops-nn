/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_proto.h
 * \brief Operator prototype definition for SwiGLU Group Dynamic Quant
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_QUANT_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_QUANT_PROTO_H_

#include "graph/operator_reg.h"

namespace ge{
    /**
    * @brief Fused SwiGLU activation with group dynamic quantization.

    * @par Inputs:
    * Four inputs, including:
    * @li x: A tensor of shape [tokens, 2H] or [B, S, 2H]. Type is float32, float16, bfloat16.
    * @li weight: An optional tensor of shape [tokens, 1]. Type is float32.
    * @li group_index: An optional tensor of shape [num_groups]. Type is int64.
    * @li scale: An optional tensor of shape [1] or [num_groups]. Type is float32.

    * @par Outputs:
    * Three outputs, including:
    * @li y: A tensor of shape [tokens, H]. Type is hifloat8.
    * @li y_scale: A tensor of shape [1] or [num_groups]. Type is float32.
    * @li y_origin: An optional tensor of shape [tokens, H]. Type matches x (output when output_origin=true).

    * @par Attributes:
    * Seven attributes, including:
    * @li dst_type: An int. Target quantization dtype, default 27 (hifloat8).
    * @li quant_mode: An int. Quantization mode, default 0.
    * @li block_size: An int. Block size, default 0.
    * @li round_scale: A bool. Scale rounding optimization, default false.
    * @li clamp_limit: A float. Clamp threshold, default 0.0 (no clamp).
    * @li dst_type_max_finite: A float. Max finite value of target quantization type, default 448.0.
    * @li output_origin: A bool. Whether to output y_origin, default false.

    * @attention Constraints:
    * The last dimension of x must be even (2H). The last dimension of y is H = 2H/2.
    * y and y_scale are required outputs; y_origin is optional (output when output_origin=true).
    */
    REG_OP(SwigluGroupQuant)
        .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
        .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
        .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_HIFLOAT8}))
        .OUTPUT(y_scale, TensorType({DT_FLOAT}))
        .OUTPUT(y_origin, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
        .ATTR(dst_type, Int, 27)
        .ATTR(quant_mode, Int, 0)
        .ATTR(block_size, Int, 0)
        .ATTR(round_scale, Bool, false)
        .ATTR(clamp_limit, Float, 0.0)
        .ATTR(dst_type_max_finite, Float, 448.0)
        .ATTR(output_origin, Bool, false)
        .OP_END_FACTORY_REG(SwigluGroupQuant)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GROUP_QUANT_PROTO_H_
