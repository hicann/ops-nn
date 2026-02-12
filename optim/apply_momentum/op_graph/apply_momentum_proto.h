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
 * \file apply_momentum_proto.h
 * \brief
 */
#ifndef OPS_NN_OPTIM_APPLY_MOMENTUM_PROTO_H
#define OPS_NN_OPTIM_APPLY_MOMENTUM_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Updates "var" according to the momentum scheme. Set use_nesterov = True if you
*  want to use Nesterov momentum. \n
*  computing process:
*@code{.c}
*  accum = accum * momentum + grad
*  if use_nesterov {
*      var -= grad * lr + accum * momentum * lr
*  } else {
*      var -= lr * accum
*  }
*@endcode
*
*@par Inputs:
*@li var: A mutable tensor of ND. Must be of dtype float16, float32 or bfloat16.
*    Specifying parameters to be updated.
*    Should be from a Variable().
*@li accum: A mutable tensor of ND. Must be of the same shape and dtype as "var".
*    Specifying gradient accumulation.
*    Should be from a Variable().
*@li lr: A scalar. Must be of the same type as "var". Specifying the learning rate.
*@li grad: A tensor of ND. Must be of the same shape and type as "var". Specifying the gradient.
*@li momentum: A scalar. Must be of the same type as "var". Specifying the momentum.

*@par Attributes:
*@li use_nesterov: An optional bool. Defaults to "False".
*    If "True", the tensor var will be updated as Nesterov momentum.
*
*@li use_locking: An optional bool. Defaults to "False".
*    If "True", updating of the "var" and "accum" tensors is protected by a lock;
*    otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same shape, dtype and format as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyMomentum.
*
*/

REG_OP(ApplyMomentum)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(momentum, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_nesterov, Bool, false)
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyMomentum)
} // namespace ge

#endif // OPS_NN_OPTIM_APPLY_MOMENTUM_PROTO_H
