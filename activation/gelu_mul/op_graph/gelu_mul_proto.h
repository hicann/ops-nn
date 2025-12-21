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
 * \file nn_activation.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_

#include "graph/operator_reg.h"

namespace ge{
/**
    * @brief GeluMul divides the input tensor into left and right tensors x1 and x2 based on the last dimension,
    * performs GELU calculation on x1 on the left, and multiplies the calculation result by x2. \n

    * @par Inputs:
    * x: A tensor of type float, float16 or bfloat16. Shape support 2D ~ 8D.
    * The format must be ND.

    * @par Attributes:
    * approximate: A optional string. The GELU approximation algorithm to use: 'none' or 'tanh', default is 'none'.

    * @par Outputs:
    * y: A tensor has the same type and format as "x".
    * Other dimensions of its shape are the same as those of "x".
    * The value of the last dimension is half the value of the last dimension of "x". \n
    */
    REG_OP(GeluMul)
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(approximate, String, "none")
        .OP_END_FACTORY_REG(GeluMul)

}
#endif  // OPS_BUILT_IN_OP_PROTO_INC_NN_ACTIVATION_H_
