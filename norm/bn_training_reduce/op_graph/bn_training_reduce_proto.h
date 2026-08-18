/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BN_TRAINING_REDUCE_PROTO_H_
#define OPS_BN_TRAINING_REDUCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* x: A 4D tensor of type float16 or float32 or bfloat16, with format NHWC or NCHW.
* Indicates the input tensor, that is, the original data to be normalized.

* @par Outputs:
* @li sum: A 1D tensor of type float32 for SUM reduced "x". It represents the sum of the input tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x".
* @li square_sum: A 1D tensor of type float32 for SUMSQ reduced "x". It represents the sum of squares of the input
tensor "x" on the C axis.
* The shape of sum is consistent with the C axis of "x". Has the same format as "x". \n

* @attention Constraints:
* This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BNTrainingReduce.
*/
REG_OP(BNTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingReduce)
} // namespace ge

#endif // OPS_BN_TRAINING_REDUCE_PROTO_H_
