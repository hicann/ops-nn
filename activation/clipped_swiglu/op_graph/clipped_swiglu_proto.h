/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file clipped_swiglu_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_PROTO_H_

#include "graph/operator_reg.h"

namespace ge{
    /**
    * @brief Activation function of SwiGlu with clipping.

    * @par Inputs:
    * Two inputs, including:
    * @li x: A tensor. Type is bfloat16, float16, float32.
    * @li group_index: An optional tensor. Shape is (N,). Type is int64. 

    * @par Outputs:
    * one output, including:
    * y: A tensor. Type is bfloat16, float16, float32.

    * @par Attributes:
    * Five attributes, including:
    * @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
    * @li alpha: An optional float. The activation coefficient for the GLU activation function, default is 1.702.
    * @li limit: An optional float. The threshold limit for SWIGLU input, default is 7.0.
    * @li bias: An optional float. The bias applied during SWIGLU linear computation, default is 1.0.
    * @li interleaved: An optional bool. The way of splitting x: true for interleaved splitting, false for front-back splitting, default is true.

    * @attention Constraints:
    * The dim dimension of x must be divisible by 2, and the dim dimension of y must be equal to the dim dimension of x divided by 2.
    */
    REG_OP(ClippedSwiglu)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(dim, Int, -1)
        .ATTR(alpha, Float, 1.702)
        .ATTR(limit, Float, 7.0)
        .ATTR(bias, Float, 1.0)
        .ATTR(interleaved, Bool, true)
        .OP_END_FACTORY_REG(ClippedSwiglu)
}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_PROTO_H_
