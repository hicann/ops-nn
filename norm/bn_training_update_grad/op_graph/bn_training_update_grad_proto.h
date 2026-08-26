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
 * \file bn_training_update_grad_proto.h
 * \brief BNTrainingUpdateGrad proto 定义（注册体与 canndev built-in 逐字一致：
 *        ops/built-in/op_proto/inc/reduce_ops.h 中的 REG_OP(BNTrainingUpdateGrad)）
 */
#ifndef OPS_NORM_BN_TRAINING_UPDATE_GRAD_PROTO_H_
#define OPS_NORM_BN_TRAINING_UPDATE_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Four inputs, including:
* @li grads: A 4D tensor of type float16 or float32 or bfloat16,
* for the gradient, with format NHWC or NCHW.
* Indicates the gradient of the loss function with respect to the output of the batch normalization layer.
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* Indicates the data input to the batch normalization layer during the forward propagation process.
* Has the same type, format and shape as "grads".
* @li batch_mean: A 1D tensor of type float32,
* for the mean of "x". Shape must be C channel.
* Has the same format as "grads".
* @li batch_variance: A 1D tensor of type float32,
* for the variance of "x" . Shape must be C channel.
* Has the same format as "grads". \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001".
* Represents a very small positive number that is added to the variance of "x" to prevent division by zero. \n

* @par Outputs:
* @li diff_scale: A 1D Tensor of type float32,
* for the offset of "scale". Shape must be C channel.
* Has the same format as "grads".

* @li diff_offset: A 1D Tensor of type float32,
* for the offset of "offset". Shape must be C channel.
* Has the same format as "grads". \n

*/
REG_OP(BNTrainingUpdateGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .ATTR(epsilon, Float, 0.0001)
    .OUTPUT(diff_scale, TensorType({DT_FLOAT}))
    .OUTPUT(diff_offset, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateGrad)

} // namespace ge

#endif // OPS_NORM_BN_TRAINING_UPDATE_GRAD_PROTO_H_
