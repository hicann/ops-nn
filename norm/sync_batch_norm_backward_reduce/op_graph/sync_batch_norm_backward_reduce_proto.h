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
 * \file sync_batch_norm_backward_reduce_proto.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_BACKWARD_REDUCE_PROTO_H
#define SYNC_BATCH_NORM_BACKWARD_REDUCE_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief part of SyncBatchNormBackward .

*@par Inputs:
* Four inputs, including:
*@li sum_dy: A ND tensor. Represents the sum of the gradients of the loss function output to the batch normalization layer.
* Must be one of the following types: float16, float32, bfloat16.
*@li sum_dy_dx_pad: A ND tensor. Represents the sum of the gradients of the input of the loss function to the batch normalization layer.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy".
*@li mean: A ND tensor. The mean value of the input data calculated during forward propagation.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy".
*@li invert_std: A ND tensor. The reciprocal of the input data standard deviation calculated during forward propagation.
* Must be one of the following types: float16, float32, bfloat16. Has the same type, shape and format as "sum_dy". \n

*@par Outputs:
*@li sum_dy_xmu: A ND tensor. Represents the sum of the gradients of the loss function against the mean.
* Has the same type, shape and format as "sum_dy".
*@li y: A ND tensor. Indicates the adjusted gradient, which is used for backpropagation to the previous layer.
* Has the same type, shape and format as "sum_dy". \n
*/
REG_OP(SyncBatchNormBackwardReduce)
    .INPUT(sum_dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum_dy_dx_pad, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(mean, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(invert_std, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum_dy_xmu, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OP_END_FACTORY_REG(SyncBatchNormBackwardReduce)
}  // namespace ge

#endif  // SYNC_BATCH_NORM_BACKWARD_REDUCE_PROTO_H
