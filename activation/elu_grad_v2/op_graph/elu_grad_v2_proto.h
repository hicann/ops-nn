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
 * \file elu_grad_v2_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_V2_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_V2_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Calculate the elu_grad_v2 function.
*Applies the element-wise function:
*Computes the backward for the elu.
*
*@par Inputs:
*Two inputs, including:
* @li grads: A tensor. Must be one of the following types:
*     float16, float32, bfloat16.
* @li activations: A tensor. Must be one of the following types:
*     float16, float32, bfloat16.
*
*@par Outputs:
*y: A Tensor with the same type and shape of grads's.
*
*@par Attributes:
* @li alpha: scalar parameter of type float32, the same as elu forward function, 
*     it is a hyperparameter controls the value to which an ELU saturates for negative net input, 
*     default value = 1.0 .
* @li scale: scalar parameter of type float32, the same scale as elu forward function scale, 
*     default value = 1.0 .
* @li input_scale: scalar parameter of type float32, which adjusts the output when activations are less than zero
*     default value = 1.0 .
* @li is_result: optional bool, if true, activations is result tensor, 
*     if false, activations is self tensor, default value is false.
*
*@attention Constraints:
*Shapes and datatypes of grads and activations must be the same.
*/
REG_OP(EluGradV2)
    .INPUT(grads, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(activations, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(alpha, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(input_scale, Float, 1.0)
    .ATTR(is_result, Bool, false)
    .OP_END_FACTORY_REG(EluGradV2)
} // namespace ge
#endif  // OPS_BUILT_IN_OP_PROTO_INC_ELU_GRAD_V2_OPS_H_
