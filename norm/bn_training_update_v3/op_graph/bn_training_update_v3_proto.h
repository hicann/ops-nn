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
 * \file bn_training_update_v3_proto.h
 * \brief BNTrainingUpdateV3 proto 定义（注册体与 canndev built-in 逐字一致：
 *        ops/built-in/op_proto/inc/reduce_ops.h 中的 REG_OP(BNTrainingUpdateV3)）
 */
#ifndef OPS_NORM_BN_TRAINING_UPDATE_V3_PROTO_H_
#define OPS_NORM_BN_TRAINING_UPDATE_V3_PROTO_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Performs reduced batch normalization. For some scenes which don't
* contain assign moving average .

* @par Inputs:
* Five inputs, including:
* @li x: A 4D tensor of type float16/float32/bfloat16, with format NHWC or NCHW. Empty tensors are not supported.
* Input tensor, that is, the original data that needs to be normalized.
* @li sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
BNTrainingReduce.
* It represents the sum of the input tensor "x" on the C axis. Has the same format as "x".
* @li square_sum: A 1D tensor of type float32, the shape is same as dim C of input "x", for the output of operator
BNTrainingReduce.
* It represents the sum of squares of the input tensor "x" on the C axis. Has the same format as "x".
* @li scale: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling factor. Has the same
format as "x".
* @li offset: A 1D tensor of type float32, the shape is same as dim C of input "x", for the scaling offset. Has the same
format as "x". \n

* @par Attributes:
* epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero. \n

* @par Outputs:
* Five outputs, including:
* @li y: A 4D tensor of type float16 or float32 or bfloat16, for normalized "x". Empty tensors are not supported.
* Has the same dype, format and shape as "x".
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel. Has the same format as "x".
* @li batch_variance: A 1D tensor of type float32, for the variance of "x". shape must be C channel. Has the same
format as "x".
* @li reserve_1: A 1D tensor of type float32, for the mean of batch "x".
* Has the same type, shape and format as input "sum".
* @li reserve_2: A 1D tensor of type float32, for the variance of batch "x".
* Has the same type, shape and format as input "sum". \n

* @attention Constraints:
* @li This operator is used in conjunction with BNTrainingReduce.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to
* the square root instruction.
* @li The 4D NHWC/NCHW description above applies to the A2-series products. For Ascend 950PR/
* Ascend 950DT: "x" supports ND (or NCHW tag, layout-normalized) with rank 2~8 ([N, C, R...],
* dim0=N, dim1=C); NHWC/NC1HWC0/NDC1HWC0 origin formats are rejected by tiling. The statistics
* inputs/outputs hold element count C (recommended shape [C]; output shapes follow "scale").
* Optional A2 attributes before_split_ori_shape/before_split_ori_format (BN FFTS scenario) are
* not registered on Ascend 950PR/Ascend 950DT and have no effect there.
*/
#ifndef OPS_PROTO_DEF_BNTRAININGUPDATEV3
#define OPS_PROTO_DEF_BNTRAININGUPDATEV3
REG_OP(BNTrainingUpdateV3)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(epsilon, Float)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_1, TensorType({DT_FLOAT}))
    .OUTPUT(reserve_2, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(BNTrainingUpdateV3)
#endif // OPS_PROTO_DEF_BNTRAININGUPDATEV3

} // namespace ge

#endif // OPS_NORM_BN_TRAINING_UPDATE_V3_PROTO_H_
