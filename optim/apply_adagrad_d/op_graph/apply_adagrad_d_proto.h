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
 * \file nn_training_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_

#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Updates "var" according to the adagrad scheme.
*   accum += grad * grad
*   var -= lr * grad * (1 / sqrt(accum))
*
*@attention Constraints:
*@li The input and output tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable(). Support float16, bfloat16 and float32.
*@li accum: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
*@li lr: A scalar. Has the same type as "var".
*@li grad: A tensor for the gradient. Has the same type as "var".
*
*@par Attributes:
*@li update_slots: An optional bool. Defaults to "True". If "True", the accum tensor will be updated.
*@li use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
*@li var: A mutable tensor. Has the same type as input "var".
*@li accum: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyAdagrad.
*
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use ApplyAdagrad instead.
*/
REG_OP(ApplyAdagradD)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .OUTPUT(accum, TensorType::NumberType())
    .ATTR(update_slots, Bool, true)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdagradD)
}// namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_NN_TRAINING_OPS_H_