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
 * \file sync_batch_norm_gather_stats_proto.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_GATHER_STATS_PROTO_H
#define SYNC_BATCH_NORM_GATHER_STATS_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief After the sum(total_sum) and the square sum(total_square_sum) are separately calculated on each device,
* a total mean(batch_mean) and reciprocal of standard deviation(batch_invstd) are returned,
* running_mean and running_var are updated.

* @par Inputs:
* include:
* @li total_sum: A 2-D tensor, that is, [N, C]. The sum of each device. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32.
* @li total_square_sum: A 2-D tensor, that is, [N, C]. The format must be ND. The square sum of each device.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as "total_sum".
* @li sample_count: A 1-D tensor. Number of data for each device. The format must be ND.
* Must be one of the following types: int32. The value of "sample_count" needs to be consistent with the N-axis value of "total_sum".
* @li mean: A 1-D tensor. Runtime mean. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32. The value of "mean" needs to be consistent with the C-axis value of "total_sum".
* @li variance: A 1-D tensor. Runtime variance. The format must be ND.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* The value of "variance" needs to be consistent with the C-axis value of "total_sum". \n

* @par Attributes:
* Two Attributes, including:
* @li momentum: An optional float. Control the update speed of the moving average. Defaults to 0.1. \n
* @li eps: An optional float. A very small value to prevent division by zero. Defaults to 0.00001. \n

* @par Outputs:
* include:
* @li batch_mean: A 1-D tensor. Total mean of this batch.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li batch_invstd: A 1-D tensor. General statistics.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li mean: A 1-D tensor. Updated Runtime mean.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean".
* @li variance: A 1-D tensor. Updated Runtime variance.
* Must be one of the following types: bfloat16, float16, float32. Has the same type, shape and format as input "mean". \n
*/
REG_OP(SyncBatchNormGatherStats)
    .INPUT(total_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(total_square_sum, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(sample_count, TensorType({DT_INT32}))
    .INPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .INPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(batch_mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(batch_invstd, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(mean, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(variance, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .ATTR(momentum, Float, 0.1f)
    .ATTR(eps, Float, 0.00001f)
    .OP_END_FACTORY_REG(SyncBatchNormGatherStats)
}  // namespace ge

#endif  // SYNC_BATCH_NORM_GATHER_STATS_PROTO_H
