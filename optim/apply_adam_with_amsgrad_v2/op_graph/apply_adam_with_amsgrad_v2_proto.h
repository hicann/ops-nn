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
 * \file apply_adam_with_amsgrad_v2_proto.h
 * \brief ApplyAdamWithAmsgradV2 GE IR REG_OP 定义
 *
 * 输入输出契约：
 * - 输入 11 个：var/m/v/vhat（state）+ beta1_power/beta2_power/lr/beta1/beta2/epsilon（标量）+ grad
 * - 输出 4 个：var/m/v/vhat（与输入 var/m/v/vhat inplace 同名）
 * - 属性 1 个（use_locking，语义占位，默认 false，不参与计算）
 */
#ifndef OPS_OP_PROTO_INC_APPLY_ADAM_WITH_AMSGRAD_V2_H_
#define OPS_OP_PROTO_INC_APPLY_ADAM_WITH_AMSGRAD_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 *@brief Updates '*var' according to the Adam algorithm..
 *   lr_t := {learning_rate} * sqrt{1 - beta_2^t} / (1 - beta_1^t)
 *   m_t := beta_1 * m_{t-1} + (1 - beta_1) * g
 *   v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g
 *   vhat_t := max{vhat_{t-1}, v_t}
 *   variable := variable - lr_t * m_t / (sqrt{vhat_t} + epsilon)
 *
 *@par Inputs:
 *Eleven inputs, including:
 *@li var: A mutable tensor of type float32 (DT_FLOAT only). Should be from a
 *    Variable().
 *@li m: A mutable tensor. Has the same type as "var". Should be from a
 *    Variable().
 *@li v: A mutable tensor. Has the same type as "var". Should be from a
 *    Variable().
 *@li vhat: A mutable tensor. Has the same type as "var". Should be from a
 *    Variable().
 *@li beta1_power: A mutable tensor. Has the same type as "var". Should be from a
 *    Variable().
 *@li beta2_power: A mutable tensor. Has the same type as "var". Should be from a
 *    Variable().
 *@li lr: A tensor for the learning rate. Has the same type as "var". Should be
 *    from a Variable().
 *@li beta1: A mutable tensor. Has the same type as "var". Should be
 *    from a Variable().
 *@li beta2: A mutable tensor. Has the same type as "var". Should be
 *    from a Variable().
 *@li epsilon: A mutable tensor. Has the same type as "var". Should be
 *    from a Variable().
 *@li grad: A tensor for the gradient. Has the same type as "var". Should be
 *    from a Variable().
 *
 *@par Attribute:
 *one attribute, including:
 *@li use_locking: An optional bool. Defaults to "False".
 *    If "True", updating of the "var" tensor is protected by a lock;
 *    otherwise the behavior is undefined, but may exhibit less contention.
 *
 *@par Outputs:
 *four outputs, including:
 *@li var: A mutable tensor. Has the same type as input "var".
 *@li m: A mutable tensor. Has the same type as input "var"
 *@li v: A mutable tensor. Has the same type as input "var"
 *@li vhat: A mutable tensor. Has the same type as input "var"
 *
 *@attention Constraints:
 * The input tensors must have the same shape.
 *
 *@par Third-party framework compatibility
 * Compatible with the TensorFlow operator ResourceApplyKerasMomentum.
 *
 */
REG_OP(ApplyAdamWithAmsgradV2)
    .INPUT(var, TensorType({DT_FLOAT}))
    .INPUT(m, TensorType({DT_FLOAT}))
    .INPUT(v, TensorType({DT_FLOAT}))
    .INPUT(vhat, TensorType({DT_FLOAT}))
    .INPUT(beta1_power, TensorType({DT_FLOAT}))
    .INPUT(beta2_power, TensorType({DT_FLOAT}))
    .INPUT(lr, TensorType({DT_FLOAT}))
    .INPUT(beta1, TensorType({DT_FLOAT}))
    .INPUT(beta2, TensorType({DT_FLOAT}))
    .INPUT(epsilon, TensorType({DT_FLOAT}))
    .INPUT(grad, TensorType({DT_FLOAT}))
    .OUTPUT(var, TensorType({DT_FLOAT}))
    .OUTPUT(m, TensorType({DT_FLOAT}))
    .OUTPUT(v, TensorType({DT_FLOAT}))
    .OUTPUT(vhat, TensorType({DT_FLOAT}))
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWithAmsgradV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_APPLY_ADAM_WITH_AMSGRAD_V2_H_
