/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_adam_w_v2_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_IS_APPLY_ADAM_W_V2_H_
#define OPS_OP_PROTO_INC_IS_APPLY_ADAM_W_V2_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
* @brief Updates "var" "m" "v" and "max_grad_norm" according to the AdamWV2 algorithm.
*
* @attention Constraints:
*  The input tensors must have the same shape, except for the step. The shape of step must be (1,).
*  When the data types of the input tensors var,m,v,grad,max_grad_norm are the same, the data type can be
*  float16,bfloat16 or float32.
*  The data types of the input tensors var,m and v must be the same.For example,if var tensor is float16,
*  the data types of m and v must also be float16.
*  The data tytpes of the input tensors grad and max_grad_norm must be the same.For example,if grad tensor is float16,
*  the data types of max_grad_norm must also be float16.
*  When data type of the input tensor var,m and v are different with input tensor grad and max_grad_norm,
*  the data types of var,m and v can only be float32,and the data type of grad and max_grad_norm tensor
*  can only be float16 or bfloat16.
*
* @par Inputs:
* @li var: A Tensor, dtype is float16 bfloat16 or float32, default is ND.
* @li m: A Tensor of the same type as "var", default is ND.
* @li v: A Tensor of the same type as "var", default is ND.
* @li grad: A Tensor, dtype is float16 bfloat16 or float32, for the gradient, default is ND.
* @li step: A Tensor, dtype is float32 or int64.
* @li max_grad_norm: An optional Tensor of the same type as "grad", default is ND.
*
* @par Attributes:
* @li lr: A required float, default is 0.1
* @li beta1: A required float, default is 0.1
* @li beta2: A required float,default is 0.1
* @li weight_decay: A required float, default is 0.1
* @li eps: A required float, default is 0.1
* @li amsgrad: A required bool, default is false.
* @li maximize: A required bool, default is false.\n
*/
REG_OP(ApplyAdamWV2)
    .INPUT(var, TensorType::FLOAT())
    .INPUT(m, TensorType::FLOAT())
    .INPUT(v, TensorType::FLOAT())
    .INPUT(grad, TensorType::FLOAT())
    .INPUT(step, TensorType({DT_FLOAT, DT_INT64}))
    .OPTIONAL_INPUT(max_grad_norm, TensorType::FLOAT())
    .ATTR(lr, Float, 0.1f)
    .ATTR(beta1, Float, 0.1f)
    .ATTR(beta2, Float, 0.1f)
    .ATTR(weight_decay, Float, 0.1f)
    .ATTR(eps, Float, 0.1f)
    .ATTR(amsgrad, Bool, false)
    .ATTR(maximize, Bool, false)
    .OP_END_FACTORY_REG(ApplyAdamWV2)

} // namespace ge

#endif // OPS_OP_PROTO_INC_IS_APPLY_ADAM_W_V2_H_
