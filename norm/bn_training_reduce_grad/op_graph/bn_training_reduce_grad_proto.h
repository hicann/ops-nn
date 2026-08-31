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
 * \file bn_training_reduce_grad_proto.h
 * \brief BNTrainingReduceGrad proto 定义（注册体与 canndev built-in 逐字一致：
 *        ops/built-in/op_proto/inc/reduce_ops.h 中的 REG_OP(BNTrainingReduceGrad)；
 *        OPS_PROTO_DEF_* 隔离宏防与 canndev built-in 同名注册源冲突）
 */
#ifndef OPS_NORM_BN_TRAINING_REDUCE_GRAD_PROTO_H_
#define OPS_NORM_BN_TRAINING_REDUCE_GRAD_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Seven inputs, including:
* @li grads: A 4D tensor of type float16 or float32 or bfloat16, for the gradient, with format NHWC or NCHW.
* The gradient of the loss function with respect to the output of the batch normalization layer.
* @li x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* It represents the data input to the batch normalization layer during the forward pass.
* Has the same type, format and shape as "grads".
* @li diff_scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
* Indicates the gradient of the loss function to the scaling parameter "scale".
* Has the same format as "grads".
* @li diff_offset: A 1D tensor of type float32, the shape is same as dim C of input grads.
* Represents the gradient of the loss function to the offset parameter.
* Has the same format as "grads".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input grads.
* The scaling parameter in batch normalization, used to adjust the normalized output.
* Has the same format as "grads".
* @li batch_mean: A 1D tensor of type float32, the shape is same as dim C of input grads, for the mean of "x".
* Has the same format as "grads".
* @li batch_variance: A 1D tensor of type float32, the shape is same as dim C of input grads, for the variance of "x".
* Has the same format as "grads". \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001".
* Represents a small positive number added to the variance of "x" to prevent division by zero. \n

* @par Outputs:
* y: A Tensor of type float16, float32 or bfloat16, with format NHWC or NCHW.
* It represents the gradient of the loss function with respect to the input data x.
* Has the same type, format and shape as "grads". \n

* @attention Constraints:
* The preceding layer of this operator must be BNTrainingUpdateGrad . \n

* @see BNTrainingUpdateGrad
*/
REG_OP(BNTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BNTrainingReduceGrad)

} // namespace ge

#endif // OPS_NORM_BN_TRAINING_REDUCE_GRAD_PROTO_H_
