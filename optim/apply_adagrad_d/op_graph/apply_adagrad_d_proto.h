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
 * \file apply_adagrad_d_proto.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_

#include "graph/operator_reg.h"
namespace ge {
/**
 *@brief Updates "var" according to the adagrad scheme.
 *   accum += grad * grad
 *   var -= lr * grad * (1 / sqrt(accum))
 *
 *@attention Constraints:
 *@li The input tensors var, accum and grad must have the same shape and dtype.
 *@li The input tensor lr must be a scalar.
 *
 *@par Inputs:
 *@li var: A mutable tensor. Should be from a Variable(). Support float16, bfloat16 and float32.
 *@li accum: A mutable tensor. Has the same type and shape as "var".
 *     Should be from a Variable().
 *@li lr: A scalar. Has the same type as "var".
 *@li grad: A tensor for the gradient. Has the same type and shape as "var".
 *
 *@par Attributes:
 *@li update_slots: An optional bool. Defaults to "true". If "true", the accum tensor will be updated;
 *     otherwise, the accum output equals the accum input.
 *@li use_locking: An optional bool. Defaults to "false".
 *     If "true", updating of the "var" and "accum" tensors is protected
 *     by a lock; otherwise the behavior is undefined, but may exhibit less
 *     contention.
 *
 *@par Outputs:
 *@li var: A mutable tensor. Has the same type and shape as input "var".
 *@li accum: A mutable tensor. Has the same type and shape as input "var".
 *
 *@par Third-party framework compatibility
 *Compatible with the TensorFlow operator ApplyAdagrad.
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
} // namespace ge
#endif // OPS_BUILT_IN_OP_PROTO_INC_APPLY_ADAGRAD_D_H_
