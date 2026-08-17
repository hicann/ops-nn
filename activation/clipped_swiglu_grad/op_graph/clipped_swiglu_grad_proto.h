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
 * \file clipped_swiglu_grad_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Compute the ClippedSwigluGrad,
* where the activations function in GLU is SwishGrad.

* @par Inputs:
* Three inputs, including:
* @li grad_y: A Tensor, which is the output gradient of forward operator and which
* has the same shape as "x" except for the dimension specified by the "dim" parameter.
* The dimension size specified by "dim" is half of the corresponding dimension of x.
* Must be one of the following types: bfloat16, float16, float32.
* @li x: A Tensor. Must be one of the following types: bfloat16, float16, float32.
* @li group_index: An optional tensor. Shape is (N,). Type is int64.

* @par Outputs:
* one Output, including:
* grad_x: A Tensor, which is the gradient of x and has the same shape as "x".
* Must be one of the following types: bfloat16, float16, float32.

* @par Attributes:
* Five attributes, including:
* @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
* @li alpha: An optional float. The activation coefficient for the GLU activation function, default is 1.702.
* @li limit: An optional float. The threshold limit for SWIGLU input, default is 7.0.
* @li bias: An optional float. The bias applied during SWIGLU linear computation, default is 1.0.
 * @li interleaved: An optional bool. The way of splitting x: true for interleaved splitting,
 * false for front-back splitting, default is true.
* @par Attention Constraints:
* The dim dimension of x must be divisible by 2, and the dim dimension of grad_y must be equal
* to the dim dimension of x divided by 2.
*/
REG_OP(ClippedSwigluGrad)
    .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(group_index, TensorType({DT_INT64}))
    .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(dim, Int, -1)
    .ATTR(alpha, Float, 1.702)
    .ATTR(limit, Float, 7.0)
    .ATTR(bias, Float, 1.0)
    .ATTR(interleaved, Bool, true)
    .OP_END_FACTORY_REG(ClippedSwigluGrad)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_CLIPPED_SWIGLU_GRAD_PROTO_H_
