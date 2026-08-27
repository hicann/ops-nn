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
 * \file bn3d_training_reduce_proto.h
 * \brief
 */
#ifndef OPS_NORM_BN3D_TRAINING_REDUCE_PROTO_H_
#define OPS_NORM_BN3D_TRAINING_REDUCE_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* x: A tensor of type float16, float32, or bfloat16. NCDHW supports rank 2 to 5, NDHWC supports rank 5,
* and NDC1HWC0 supports rank 6 on applicable products.
* When the C axis is 0, other dimensions support empty tensors; when the C axis is not 0, other dimensions do not
* support empty tensors.

* @par Outputs:
* @li sum: A tensor of type float32 for SUM reduced "x". Its shape is [C] for NCDHW and NDHWC, and
* [1, 1, C1, 1, 1, C0] for NDC1HWC0.
* @li square_sum: A tensor of type float32 for SUMSQ reduced "x". It has the same shape and format as "sum". \n

* @attention Constraints:
* This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BN3DTrainingUpdate.
*/
#ifndef OPS_PROTO_DEF_BN3DTRAININGREDUCE
#define OPS_PROTO_DEF_BN3DTRAININGREDUCE
REG_OP(BN3DTrainingReduce)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(sum, TensorType({DT_FLOAT}))
    .OUTPUT(square_sum, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BN3DTrainingReduce)
#endif // OPS_PROTO_DEF_BN3DTRAININGREDUCE
} // namespace ge

#endif // OPS_NORM_BN3D_TRAINING_REDUCE_PROTO_H_
