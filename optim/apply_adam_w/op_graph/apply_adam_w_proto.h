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
 * \file apply_adam_w_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_EXPERIMENT_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
* @brief Updates "var" according to the AdamW algorithm. For clarity, the output variable is suffixed with "_out" in the fomulas below (e.g. m_out).
* @code{.c}
*  if maximize:
*       gt = -grad
*   else:
*       gt = grad
*   m_out = m * beta1 - (beta1 - 1) * gt
*   v_out = v * beta2 - (beta2 - 1) * gt * gt
*   var_t = var * (1 + (-lr * weight_decay))
*   beta1_power_out = beta1_power * beta1
*   beta2_power_out = beta2_power * beta2
*   if amsgrad:
*       max_grad_norm_out = max(max_grad_norm, v_out)
*       denom = sqrt(-max_grad_norm_out / (beta2_power_out -1)) + epsilon
*   else:
*       denom = sqrt(-v_out / (beta2_power_out -1)) + epsilon
*   var_out = var_t + ((lr / (beta1_power_out -1)) * (m_out / denom))
* @endcode
*
* @par Inputs:
* @li var: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying parameters to be updated.
*     Should be from a Variable().
* @li m: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying first moment.
*     Should be from a Variable().
* @li v: A tensor of ND, dtype is float16 bfloat16 or float32.
*     Specifying second moment. Values must be greater than or equal to 0.
*     Should be from a Variable().
* @li beta1_power: A scalar of the same type as "var", value is beta1^(step-1), between 0 and 1.
* @li beta2_power: A scalar of the same type as "var", value is beta2^(step-1), between 0 and 1.
* @li lr: Specifying learning_rate. A scalar of the same type as "var", value is between 0 and 1 generally (e.g. 1e-3).
* @li weight_decay: Specifying weight decay. A scalar of the same type as "var", value is between 0 and 1 generally (e.g. 1e-2).
* @li beta1: Specifying beta1. A scalar of the same type as "var", value is between 0 and 1 (e.g. 0.9).
* @li beta2: Specifying beta2. A scalar of the same type as "var", value is between 0 and 1 (e.g. 0.999).
* @li epsilon: A scalar of the same type as "var", used to improve numerical stability.
*     Should be a minimum value greater than 0 (e.g. 1e-8).
* @li grad: A tensor of the same type as "var", dtype is float16 bfloat16 or float32.
*     Specifying gradient.
* @li max_grad_norm: A mutable tensor of the same type as "var", an optional input,
*     dtype is float16 bfloat16 or float32.
*     Should be from a Variable().
*
* @par Attributes:
* @li amsgrad: An optional bool. Specifying using max_grad_norm in the calculation. Defaults to "False", only support "False".
* @li maximize: An optional bool. Specifying maximize the objective with respect to the params, instead of minimizing. Defaults to "False".
*
* @par Outputs:
* @li var: A mutable tensor. Has the same shape and type as input "var".
* @li m: A mutable tensor. Has the same shape and type as input "m".
* @li v: A mutable tensor. Has the same shape and type as input "v".
*
* @par attention Constraints:
* @li The input tensors must have the same shape.
* @li Ensure the argument for the square root is non-negative.
*/
REG_OP(ApplyAdamW)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(weight_decay, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OPTIONAL_INPUT(max_grad_norm, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamW)
}
#endif