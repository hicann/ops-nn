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
 * \file bn3_d_training_reduce_grad_proto.h
 * \brief BN3DTrainingReduceGrad 的图 IR 原型注册（REG_OP）
 */

#ifndef BN3_D_TRAINING_REDUCE_GRAD_PROTO_H
#define BN3_D_TRAINING_REDUCE_GRAD_PROTO_H

#include "graph/operator_reg.h" // REG_OP, OP_END_FACTORY_REG, TensorType macros

namespace ge {

/**
* @brief Performs the backpropagation of BatchNorm .

* @par Inputs:
* Seven inputs, including:
* @li grads: A 5Dtensor of type float16 or float32 or bfloat16, for the gradient, with format NDHWC or NCDHW.
* @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
* @li diff_scale: A 1D tensor of type float32,
* for the mean of "x". shape must be C channel.
* @li diff_offset: A 1D tensor of type float32,
* for the variance of "x". shape must be C channel.
* @li scale: A 1D tensor of type float32.
* @li batch_mean: A 1D tensor of type float32,
* for the mean of "x". shape must be C channel.
* @li batch_variance: A 1D tensor of type float32,
* for the variance of "x" . shape must be C channel. \n

* @par Attributes:
* epsilon: An optional float32. Defaults to "0.0001". A small float number
* added to the variance of "x" . \n

* @par Outputs:
* y: A 5D Tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW. \n

* @attention Constraints:
* The preceding layer of this operator must be BN3DTrainingReduceGrad . \n

* @see BN3DTrainingReduceGrad
*/
#ifndef OPS_PROTO_DEF_BN3DTRAININGREDUCEGRAD
#define OPS_PROTO_DEF_BN3DTRAININGREDUCEGRAD
REG_OP(BN3DTrainingReduceGrad)
    .INPUT(grads, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(diff_scale, TensorType({DT_FLOAT}))
    .INPUT(diff_offset, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(batch_mean, TensorType({DT_FLOAT}))
    .INPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(epsilon, Float, 0.0001)
    .OP_END_FACTORY_REG(BN3DTrainingReduceGrad)
#endif // OPS_PROTO_DEF_BN3DTRAININGREDUCEGRAD
} // namespace ge

#endif
