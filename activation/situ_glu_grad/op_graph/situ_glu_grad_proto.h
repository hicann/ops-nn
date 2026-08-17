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
 * \file situ_glu_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GRAD_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief SiTU gated linear unit backward.

* @par Inputs:
* Two inputs:
* @li grad_y: A tensor. Type is float32/float16/bfloat16. The dim dimension is half of x.
* @li x: A tensor. Type is float32/float16/bfloat16. The dim dimension must be divisible by 2.

* @par Outputs:
* One output:
* grad_x: A tensor. Type is float32/float16/bfloat16. The shape is the same as x.

* @par Attributes:
* Four attributes:
* @li dim: An optional int. The dimension to be split, value in [-xDim, xDim-1], default is -1.
* @li beta: An optional float. The scale factor for the SiTU gate activation, default is 1.0.
* @li linear_beta: An optional float. The scale factor for the linear tanh on the up path. When <= 0, the up path is
*   used as-is. default is 0.0.
* @li activate_left: An optional bool. Whether the left (front) half of x is the gate. default is true.
*/
REG_OP(SituGluGrad)
    .INPUT(grad_y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(grad_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(dim, Int, -1)
    .ATTR(beta, Float, 1.0)
    .ATTR(linear_beta, Float, 0.0)
    .ATTR(activate_left, Bool, true)
    .OP_END_FACTORY_REG(SituGluGrad)
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_SITU_GLU_GRAD_PROTO_H_
