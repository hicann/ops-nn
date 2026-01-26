/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_OP_APPLY_ADAM_D_PROTO_H
#define GE_OP_APPLY_ADAM_D_PROTO_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Updates "var" according to the Adam algorithm.
*  lr = learning_rate * (sqrt(1 - beta2_power)) / (1 - beta1_power)
*  m = m + (1 - beta1) * (grad - m)
*  v = v + (1 - beta2) * (grad * grad - v)
*  if use_nesterov == True:
*      var = var - lr * (m * beta1 + (1 - beta1) * grad) / (epsilon + sqrt(v))
*  else:
*      var = var - lr * m / (epsilon + sqrt(v))
*
*@attention Constraints:
*  *The input tensors must have the same shape.*
*
*@par Inputs:
*@li var: A mutable Tensor of the type TensorType::NumberType().
*     Should be from a Variable().
*@li m: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li v: A mutable Tensor of the same type as "var".
*     Should be from a Variable().
*@li beta1_power: A scalar of the same type as "var".
*@li beta2_power: A scalar of the same type as "var".
*@li lr: learning_rate. A scalar of the same type as "var".
*@li beta1: A scalar of the same type as "var".
*@li beta2: A scalar of the same type as "var".
*@li epsilon: A scalar of the same type as "var".
*@li grad: A Tensor of the same type as "var", for the gradient.
*
*@par Attributes:
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", m", and "v" tensors will be protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*@li use_nesterov: An optional bool. Defaults to "False".
      If "True", uses the nesterov update.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li m: A mutable tensor. Has the same type as input "m".
*@li v: A mutable tensor. Has the same type as input "v" . \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdam.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdam instead.
*/
REG_OP(ApplyAdamD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(m, TensorType::NumberType())
    .INPUT(v, TensorType::NumberType())
    .INPUT(beta1_power, TensorType::NumberType())
    .INPUT(beta2_power, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(beta1, TensorType::NumberType())
    .INPUT(beta2, TensorType::NumberType())
    .INPUT(epsilon, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(m, TensorType::NumberType())
    .OUTPUT(v, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .ATTR(use_nesterov, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamD)
}  // namespace ge

#endif  // GE_OP_APPLY_ADAM_D_PROTO_H