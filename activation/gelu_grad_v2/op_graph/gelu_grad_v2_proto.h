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
 * \file gelu_grad_v2_proto.h
 * \brief
 */
#ifndef GELU_GRAD_V2_PROTO_H_
#define GELU_GRAD_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes the gradient for the gelu of "x" .

* @par Inputs:
* Two inputs, including:
* @li dy: A Tensor. Support 1D ~ 8D. Must be one of the following types:bfloat16, float16, float32.
* @li x: A Tensor of the same type and format as "dy".

* @par Outputs:
* z: A Tensor. Has the same type, shape and format as "dy".

* @par Attributes:
* approximate: A optional string.
* The gelu grad approximation algorithm to use: 'none' or 'tanh', default is 'none'. \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator GeluGrad.

* @attention Constraints:
* if the GeluGradV2 operator has approximate='none':
* when x is -inf, the computation result is 0.
* when x is inf, the computation result is dy.

*/
REG_OP(GeluGradV2)
    .INPUT(dy, "T")
    .INPUT(x, "T")
    .OUTPUT(z, "T")
    .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(approximate, String, "none")
    .OP_END_FACTORY_REG(GeluGradV2)

}
#endif