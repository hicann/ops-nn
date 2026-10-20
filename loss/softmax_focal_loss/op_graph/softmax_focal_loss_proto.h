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
 * \file softmax_focal_loss_proto.h
 * \brief
 */
#ifndef OPS_LOSS_SOFTMAX_FOCAL_LOSS_H_
#define OPS_LOSS_SOFTMAX_FOCAL_LOSS_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {
/**
*@brief Computes the softmax focal loss.

*@par Inputs:
*Three inputs, including:
* @li pred: A Tensor. Must be one of the following types: float16, float32.
*The probabilities produced by a preceding softmax. The last axis is the class axis and the
*remaining axes are sample axes. On Ascend 950PR&950DT products any rank >= 1 is
*accepted; the other products only support the two-dimensional form "batch_size * num_classes".
* @li target: A Tensor of type int32. The one-hot ground truth, same shape as "pred".
* @li weight: An optional Tensor. Must be one of the following types: float16, float32.
*Per-element weight, same shape as "pred". Treated as all ones when absent. \n

*@par Attributes:
* @li gamma: An optional float. Exponential coefficient of the focal loss. Defaults to "2.0".
* @li alpha: An optional float. Weighted coefficient of the focal loss. Defaults to "0.25".
* @li reduction: An optional string from "none", "mean" and "sum", specifying how the loss is reduced.
* Defaults to "mean" (aligned with the common loss convention and with the Ascend 910B prototype).
* Shape of "y" follows this attribute: "none" keeps the shape of "pred"; "mean"/"sum" would reduce "y"
* to a scalar. Only "none" is implemented currently, any other value is rejected in InferShape;
* so "reduction" must be set explicitly to "none" for now.

*@par Outputs:
*y: A Tensor. Has the same type and shape as "pred". All elements within one row share the same
*value, which is the focal loss of that row. \n
*/
#ifndef OPS_PROTO_DEF_SOFTMAXFOCALLOSS
#define OPS_PROTO_DEF_SOFTMAXFOCALLOSS
REG_OP(SoftmaxFocalLoss)
    .INPUT(pred, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .ATTR(gamma, Float, 2.0)
    .ATTR(alpha, Float, 0.25)
    .ATTR(reduction, String, "mean")
    .OP_END_FACTORY_REG(SoftmaxFocalLoss)
#endif // OPS_PROTO_DEF_SOFTMAXFOCALLOSS

} // namespace ge

#endif // OPS_LOSS_SOFTMAX_FOCAL_LOSS_H_
