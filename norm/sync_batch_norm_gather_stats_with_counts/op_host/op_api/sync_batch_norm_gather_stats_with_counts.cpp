/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sync_batch_norm_gather_stats_with_counts.cpp
 * \brief
 */
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "sync_batch_norm_gather_stats_with_counts.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SyncBatchNormGatherStatsWithCounts);
const std::array<const aclTensor*, 2> SyncBatchNormGatherStatsWithCounts(
    const aclTensor* meanAll, const aclTensor* invstdAll, const aclTensor* countAll, const aclTensor* meanBroadcast,
    const aclTensor* countSum, const aclTensor* runningVar, const float momentum, const float epsilon,
    aclOpExecutor* executor)
{
    L0_DFX(
        SyncBatchNormGatherStatsWithCounts, meanAll, invstdAll, countAll, meanBroadcast, countSum, runningVar, momentum,
        epsilon);

    auto invstdOut = executor->AllocTensor(
        Shape({invstdAll->GetViewShape()[1]}), invstdAll->GetDataType(), invstdAll->GetViewFormat());
    auto runningVarOut =
        executor->AllocTensor(runningVar->GetViewShape(), runningVar->GetDataType(), runningVar->GetViewFormat());
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBatchNormGatherStatsWithCounts, OP_INPUT(meanAll, invstdAll, countAll, meanBroadcast, countSum, runningVar),
        OP_OUTPUT(invstdOut, runningVarOut), OP_ATTR(momentum, epsilon));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(
            ACLNN_ERR_INNER_NULLPTR, "SyncBatchNormGatherStatsWithCountsAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {invstdOut, runningVarOut};
}
} // namespace l0op