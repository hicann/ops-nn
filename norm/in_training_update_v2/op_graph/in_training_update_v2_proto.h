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
 * \file in_training_update_v2_proto.h
 * \brief Public IR prototype mirrored from canndev reduce_ops.h.
 */

#ifndef OPS_NORM_IN_TRAINING_UPDATE_V2_PROTO_H_
#define OPS_NORM_IN_TRAINING_UPDATE_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Performs update instance normalization. \n

 * @par Inputs:
 * Seven inputs, including:
 * @li x: A 4D tensor of type float16 or float32, format [NCHW, NHWC].
 * @li sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and HW=1.
 * @li square_sum: A 4D tensor of type float32 for the output of operator INTrainingReduceV2, format [NCHW, NHWC], and
 * HW=1.
 * @li gamma: A 4D optional tensor of type float32, for the scaling gamma, format [NCHW, NHWC], and HW=1.
 * @li beta: A 4D optional tensor of type float32, for the scaling beta, format [NCHW, NHWC], and HW=1.
 * @li mean: A 4D optional tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
 * @li variance: A 4D optional tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1.\n

 * @par Attributes:
 * @li momentum: A optional float32, specifying the momentum to update mean and var. default to 0.1.
 * @li epsilon: A optional float32, specifying the small value added to variance to avoid dividing by zero. default to
 * 0.00001. \n

 * @par Outputs:
 * Three outputs
 * @li y: A 4D tensor of type float16 or float32, for normalized "x", format [NCHW, NHWC].
 * @li batch_mean: A 4D tensor of type float32, for the updated mean, format [NCHW, NHWC], and HW=1.
 * @li batch_variance: A 4D tensor of type float32, for the updated variance, format [NCHW, NHWC], and HW=1. \n

 * @attention Constraints:
 * This operator is a InstanceNorm fusion operator for updating the moving averages for training.
 * This operator is used in conjunction with INTrainingReduceV2.
*/
#ifndef OPS_PROTO_DEF_INTRAININGUPDATEV2
#define OPS_PROTO_DEF_INTRAININGUPDATEV2
REG_OP(INTrainingUpdateV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(gamma, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(beta, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(mean, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(variance, TensorType({DT_FLOAT}))
    .ATTR(momentum, Float, 0.1)
    .ATTR(epsilon, Float, 0.00001)
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(INTrainingUpdateV2)
#endif
} // namespace ge

#endif // OPS_NORM_IN_TRAINING_UPDATE_V2_PROTO_H_
