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
 * \file kl_div_loss_grad_proto.h
 * \brief
 */
#ifndef KL_DIV_LOSS_GRAD_PROTO_H_
#define KL_DIV_LOSS_GRAD_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Computes Kl_div_loss_grad or Kl_div_loss_backward. \n

* @par Inputs:
* Three inputs, including:
* @li grad: A tensor. Must be one of the following types: float16, float32, bfloat16.
*  Shape needs to satisfy the broadcast relationship with input.
*  Required.
* @li input: A tensor. Has the same type as "grad". Required.
* @li target: A tensor. Has the same type as "grad". Required.
*  Shape needs to satisfy the broadcast relationship with input. \n

* @par Attributes:
* @li reduction: An optional attribute of type String. Defaults to "mean".
*  Supports "none" | "mean" | "sum" | "batchmean".  \n
*  'none' means no reduction should be applied.  \n
*  'mean' means the total output will be divided by the number of elements in the output.  \n
*  'sum' means the output will be summed. \n
*  'batchmean' means the total output will be divided by the number of batches. \n
* @li log_target: An optional attribute of type Bool. Defaults to false. \n

* @par Outputs:
* y: A tensor. Has the same type as "grad".
* Has the same shape as "input"  \n

* @par Third-party framework compatibility
* Compatible with the Pytorch operator KlDivLossGrad.
*/
REG_OP(KlDivLossGrad)
    .INPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(input, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(target, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(reduction, String, "mean")
    .ATTR(log_target, Bool, false)
    .OP_END_FACTORY_REG(KlDivLossGrad)
}

#endif
