/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_SYNC_BATCH_NORM_GATHER_STATS_FUSED_H
#define OP_API_INC_LEVEL0_SYNC_BATCH_NORM_GATHER_STATS_FUSED_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const std::array<const aclTensor*, 4> SyncBatchNormGatherStatsFused(
    const aclTensor* mean, const aclTensor* invstd, const aclTensor* counts, const aclTensor* runningMean,
    const aclTensor* runningVar, float momentum, float eps, aclOpExecutor* executor);

} // namespace l0op

#endif