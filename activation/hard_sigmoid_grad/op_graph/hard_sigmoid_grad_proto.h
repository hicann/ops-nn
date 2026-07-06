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
 * \file hard_sigmoid_grad_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Calculate the backward outputs of the function "hard_sigmoid"

*@par Inputs:
*Two inputs, including:
* @li grads: A ND tensor. The shape should be within the range of 0D to 8D.
*     Must be one of the following types:float16, float32, bfloat16.
* @li input_x: A ND tensor. The shape should be within the range of 0D to 8D.
*     Must be one of the following types: float16, float32, bfloat16.

*@par Outputs:
*One output, including:
* y: A ND tensor with the same type and shape as 'input_x'.

* @par Attributes:
* @li alpha: An optional float. Slope of the operator, defaults to 0.16666666.
* @li beta: An optional float. Offset of the operator, defaults to 0.5.

*@par Third-party framework compatibility
*Compatible with the Pytorch operator HardSigmoidGrad.
*/
REG_OP(HardSigmoidGrad)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(input_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 0.16666666)
    .ATTR(beta, Float, 0.5)
    .OP_END_FACTORY_REG(HardSigmoidGrad) // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_HARD_SIGMOID_GRAD_OPS_H_
