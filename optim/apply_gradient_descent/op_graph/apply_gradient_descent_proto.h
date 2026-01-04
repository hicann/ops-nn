/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_gradient_descent_proto.h
 * \brief
 */
#ifndef OPS_OPTIM_APPLY_GRADIENT_DESCENT_GRAPH_PLUGIN_PROTO_H_
#define OPS_OPTIM_APPLY_GRADIENT_DESCENT_GRAPH_PLUGIN_PROTO_H_
#include "graph/operator_reg.h"
namespace ge {
/**
*@brief Updates "var" by subtracting 'alpha' * 'delta' from it.
*   var -= delta * alpha
*
*@attention Constraints:
*  the input tensors must have the same shape.
*
*@par Inputs:
*@li var: A mutable tensor. Should be from a Variable().
*@li alpha: A scalar. Has the same type as "var".
*@li delta: A tensor for the change. Has the same type as "var".
*
*@par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
*@par Outputs:
* var: A mutable tensor. Has the same type as input "var".
*
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ApplyGradientDescent.
*
*/
REG_OP(ApplyGradientDescent)
    .INPUT(var, TensorType::NumberType())
    .INPUT(alpha, TensorType::NumberType())
    .INPUT(delta, TensorType::NumberType())
    .OUTPUT(var, TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyGradientDescent)
}  // namespace ge

#endif // OPS_OPTIM_APPLY_GRADIENT_DESCENT_GRAPH_PLUGIN_PROTO_H_