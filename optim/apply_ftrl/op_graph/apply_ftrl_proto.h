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
 * \file apply_ftrl_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_APPLY_FTRL_H_
#define OPS_OP_PROTO_INC_IS_APPLY_FTRL_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Updates "var" according to the Ftrl-proximal scheme.
*@code{.c}
* accum_new = accum + grad * grad
* linear += grad - (accum_new^(-lr_power) - accum^(-lr_power)) * var / lr
* quadratic = accum_new^(-lr_power) / lr + 2 * l2
* var = (sign(linear) * l1 - linear) / quadratic if |linear| > l1 else 0.0
* accum = accum_new
*@endcode
*
*@par Inputs:
* @li var: A tensor of ND. Must be of dtype float16, float32 or bfloat16.
*     Specifying parameters to be updated.
*     Should be from a Variable().
* @li accum: A tensor of ND. Must be of the same shape and dtype as "var".
*     Specifying gradient accumulation.
*     Should be from a Variable().
* @li linear: A tensor of ND. Must be of the same shape and dtype as "var".
*     Specifying correction item.
*     Should be from a Variable().
* @li grad: A tensor of ND. Must be of the same shape and dtype as "var", specifying gradient.
* @li lr: A scalar of the same dtype as "var", specifying the scaling factor.
* @li l1: A scalar of the same dtype as "var", specifying L1 regulariation.
* @li l2: A scalar of the same dtype as "var", specifying L2 regulariation.
* @li lr_power: A scalar of the same dtype as "var", for the scaling factor.
*
*@par Attributes:
*use_locking: An optional bool. Defaults to "False". Only support "False".
*     If "True", updating of the "var" tensors will be protected by a lock;
*     otherwise the behavior is undefined, but may exhibit less contention.
*
*@par Outputs:
*var: A mutable Tensor. Must be of the same shape and dtype as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyFtrl.
*/
REG_OP(ApplyFtrl)
    .INPUT(var, TensorType::NumberType())
    .INPUT(accum, TensorType::NumberType())
    .INPUT(linear, TensorType::NumberType())
    .INPUT(grad, TensorType::NumberType())
    .INPUT(lr, TensorType::NumberType())
    .INPUT(l1, TensorType::NumberType())
    .INPUT(l2, TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrl)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_APPLY_FTRL_H_
