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
 * \file logit_grad_proto.h
 * \brief
 */

#ifndef LOGIT_GRAD_PROTO_H
#define LOGIT_GRAD_PROTO_H
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Backpropagation of Logit. \n

*@par Inputs:
*@li x: A input tensor of type float, float16 or bfloat16. Shape support 0D ~ 8D.
* Input data for the backpropagation of the probability to logit transformation
* The format must be ND.
*@li dy: Gradient of the positive output result. A Tensor with the same type, shape, format as "x".

*@par Attributes:
*eps: The epslion of "x", an optional attribute, the type is float. Defaults to -1.0. \n

*@par Outputs:
*dx: A output ensor with the same type, shape, format as "x". Probability to logit conversion backpropagates the output
data. \n
*/

REG_OP(LogitGrad)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(dy, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(dx, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(eps, Float, -1.0)
    .OP_END_FACTORY_REG(LogitGrad)

} // namespace ge
#endif