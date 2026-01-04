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
 * \file nll_loss_grad_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_
#define OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief The negative log likelihood loss grad.

* @par Inputs:
* @li x: A 2D or 4D tensor dtype of float32 or bfloat16 or float16 with shape (N, C) or (N, C, H, W).
* @li y_grad: A 1D or 3D tensor dtype of float32 or bfloat16 or float16 with shape (N) or (N, H, W) if reduction is "none". When x is 2D(shape (N, C)), y_grad should be 1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), y_grad should be 3d(shape(N, H, W)). Otherwise, shape is (1).
* @li target: Indicates the real label. A 1D or 3D tensor dtype of int32, int64 or uint8 with shape (N) or (N, H, W). When x is 2D(shape (N, C)), target should be 1D(shape(N)) or scalar, when x is 4D(shape(N, C, H, W)), target should be 3d(shape(N, H, W)).
* @li weight: Indicates the weight of each class. A 1D tensor dtype of float32 or bfloat16 or float16 with shape (C).
* @li total_weight: A 1D tensor dtype of float32 or bfloat16 or float16 with shape (1).

* @par Attributes:
* @li reduction: Computation method of the loss function. An optional string. Defaults to "mean" .
* @li ignore_index: Specifies a target value that is ignored and does not affect the input gradient. An optional int. Defaults to -100.

* @par Outputs:
* x_grad: A tensor has the same shape as "x". Must be the following type: float32, bfloat16, float16.

* @par Third-party framework compatibility
* Compatible with pytorch NLLLossGrad operator
*/
REG_OP(NLLLossGrad)
    .INPUT(x, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(y_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(target, TensorType({DT_INT32, DT_INT64, DT_UINT8}))
    .INPUT(weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .INPUT(total_weight, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .OP_END_FACTORY_REG(NLLLossGrad)
}  // namespace ge

#endif  // OPS_OP_PROTO_INC_NLL_LOSS_GRAD_OPS_H_