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
 * \file softmax_focal_loss_grad_proto.h
 * \brief
 */
#ifndef OPS_LOSS_SOFTMAX_FOCAL_LOSS_GRAD_H_
#define OPS_LOSS_SOFTMAX_FOCAL_LOSS_GRAD_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes the gradient of the softmax focal loss.

*@par Inputs:
*Four inputs, including:
* @li pred: A Tensor. Must be one of the following types: float16, float32.
*The probabilities produced by a preceding softmax, shape "batch_size * num_classes".
* @li target: A Tensor of type int32. The one-hot ground truth, same shape as "pred".
* @li dout: A Tensor. Has the same type and shape as "pred". The incoming gradient.
* @li weight: An optional Tensor. Has the same type and shape as "pred".
*Treated as all ones when absent. \n

*@par Attributes:
* @li gamma: An optional float. Exponential coefficient of the focal loss. Defaults to "2.0".
* @li alpha: An optional float. Weighted coefficient of the focal loss. Defaults to "0.25".
* @li reduction: An optional string. Defaults to "mean". When "mean", the gradient is scaled by
*1 / numel(pred); "none" and "sum" apply no scaling. \n

*@par Outputs:
*grad: A Tensor. Has the same type and shape as "pred". \n
*/
REG_OP(SoftmaxFocalLossGrad)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(dout, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(grad, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLossGrad)

} // namespace ge

#endif // OPS_LOSS_SOFTMAX_FOCAL_LOSS_GRAD_H_
