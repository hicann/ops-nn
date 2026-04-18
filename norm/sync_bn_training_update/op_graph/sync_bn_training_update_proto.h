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
 * \file sync_bn_training_update_proto.h
 * \brief
 */
#ifndef SYNC_BN_TRAINING_UPDATE_PROTO_H
#define SYNC_BN_TRAINING_UPDATE_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief update running_mean.

* @par Inputs:
* include:
* @li mean: A Tensor. The mean of each device. Must be one of the following types: float16, float32, bfloat16.
* @li running_mean: A Tensor. Runtime Mean. Must be one of the following types: float16, float32, bfloat16. \n

* @par Attributes:
* One Attribute, including:
* @li momentum: A optional float. Defaults to 0.01. \n

* @par Outputs:
* include:
* @li running_mean_update: A Tensor. It's moving mean of each device after the update. Must be one of the following types: float16, float32, bfloat16.\n

* @par Third-party framework compatibility
* ReduceMeanWithCount and SyncBatchNormGatherStatsWithCounts and SyncBNTrainingUpdate
* compatible with the Pytorch operator BatchNormGatherStatsWithCounts.
*/
REG_OP(SyncBNTrainingUpdate)
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(running_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(running_mean_update, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(momentum, Float, 0.1f)
    .OP_END_FACTORY_REG(SyncBNTrainingUpdate)
}  // namespace ge

#endif  // SYNC_BN_TRAINING_UPDATE_PROTO_H
