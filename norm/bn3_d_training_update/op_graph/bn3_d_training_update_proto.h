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
 * \file bn3_d_training_update_proto.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_PROTO_H
#define BN3_D_TRAINING_UPDATE_PROTO_H

#include "graph/operator_reg.h" // REG_OP, TensorType, INPUT, OUTPUT, OP_END_FACTORY_REG

namespace ge {

/**
* @brief Performs reduced batch normalization .

* @par Inputs:
* Seven inputs, including:
* @li x: A 5D tensor of type float16 or float32 or bfloat16, with format NDHWC or NCDHW.
* @li sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C channel.
* @li square_sum: A 1D tensor of type float32 for the output of operator BN3DTrainingUpdate. shape must be C
channel.
* @li scale: A 1D tensor of type float32, for the scaling factor. shape must be C channel.
* @li offset: A 1D tensor of type float32, for the scaling offset. shape must be C channel.
* @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
* @li variance: A 1D tensor of type float32, for the updated variance . shape must be C channel. \n

* @par Attributes:
* @li epsilon: A required float32, specifying the small value added to variance
* to avoid dividing by zero.
* @li factor: A required float32, specifying the weight for updating the mean
* and variance . \n

* @par Outputs:
* Five outputs, including:
* @li y: A 5D tensor of type float16 or float32 or bfloat16, for normalized "x", with format NDHWC or NCDHW.
* @li mean: A 1D tensor of type float32, for the updated mean. shape must be C channel.
* @li variance: A 1D tensor of type float32, for the updated variance. shape must be C channel.
* @li batch_mean: A 1D tensor of type float32, for the mean of "x". shape must be C channel.
* @li batch_variance: A 1D tensor of type float32, for the variance of "x" . shape must be C channel. \n

* @attention Constraints:
* @li This operator is a BatchNorm fusion operator for updating the moving
* averages for training.
* This operator is used in conjunction with BN3DTrainingUpdate.
* @li For Atlas 200/300/500 Inference Product, the result accuracy fails to reach 1/1000 due to the square
* root instruction.
*/
#ifndef OPS_PROTO_DEF_BN3DTRAININGUPDATE
#define OPS_PROTO_DEF_BN3DTRAININGUPDATE
REG_OP(BN3DTrainingUpdate)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .INPUT(sum, TensorType({DT_FLOAT}))
    .INPUT(square_sum, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(offset, TensorType({DT_FLOAT}))
    .INPUT(mean, TensorType({DT_FLOAT}))
    .INPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT}))
    .OUTPUT(variance, TensorType({DT_FLOAT}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT}))
    .OUTPUT(batch_variance, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(factor, Float)
    .REQUIRED_ATTR(epsilon, Float)
    .OP_END_FACTORY_REG(BN3DTrainingUpdate)
#endif // OPS_PROTO_DEF_BN3DTRAININGUPDATE

} // namespace ge

#endif
