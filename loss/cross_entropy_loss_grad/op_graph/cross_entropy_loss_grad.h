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
 * \file cross_entropy_loss_grad.h
 * \brief
 */
#ifndef OPS_LOSS_CROSS_ENTROPY_LOSS_GRAD_H_
#define OPS_LOSS_CROSS_ENTROPY_LOSS_GRAD_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs the backpropagation of CrossEntropyLoss for training scenarios .

* @par Inputs:
* Six inputs, including:
* @li grad_loss: A 1D tensor or scalar of type float16 or float32 or bfloat16, specifying the backpropagation gradient. When reduction is "none", the shape is [N], when reduction is "sum" or "mean", it is scalar. N is batch size.
* @li log_prob: A 2D tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss, specifying the logarithmized probabilities of the outputs. Shape only support [N, C], where N is batch size, C is class.
* @li target: A 1D tensor of type int32 or int64, specifying the target value. It represent the target squences. Shape is [N]. The range of values is [0, C).
* @li weight: An optional 1D tensor of type float32, specifying the weight value. Shape is [C].
* @li grad_zloss: An optional ND tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss. Reserved.
* @li lse_gor_zloss: An optional ND tensor of type float16 or float32 or bfloat16. Has the same dtype as grad_loss. Reserved.

* @par Attributes:
* @li reduction: A character string from "none", "mean", and "sum", specifying the gradient output mode. Defaults to "mean" .
* @li ignore_index: An optional int. Specifies a target value that is ignored and does not contribute to the input gradient. Defaults to -100.
* @li label_smoothing: An optional float attr in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss. Defaults to 0.0.
* @li lse_square_scale_for_zloss: An optional float attr, Default: 0.0. Reserved.

* @par Outputs:
* x_grad: A 2D tensor. Has the same dtype and shape as log_prob.
*/
REG_OP(CrossEntropyLossGrad)
    .INPUT(grad_loss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(log_prob, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT64, DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(grad_zloss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(lse_gor_zloss, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OUTPUT(x_grad, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
    .ATTR(label_smoothing, Float, 0.0)
    .ATTR(lse_square_scale_for_zloss, Float, 0.0)
    .OP_END_FACTORY_REG(CrossEntropyLossGrad)

} // namespace ge

#endif // OPS_LOSS_CROSS_ENTROPY_LOSS_GRAD_H_