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
 * \file sync_batch_norm_gather_stats.cpp
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
#include "opdev/common_types.h"
#include "sync_batch_norm_gather_stats.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SyncBatchNormGatherStats);

const std::array<aclTensor*, SYNC_BATCH_NORM_OUTPUT_NUM> SyncBatchNormGatherStats(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, float momentum, float eps, aclOpExecutor* executor)
{
    L0_DFX(SyncBatchNormGatherStats, totalSum, totalSquareSum, sampleCount, mean, variance, momentum, eps);

    auto batchMean = executor->AllocTensor(
        mean->GetStorageShape(), mean->GetOriginalShape(), mean->GetDataType(), mean->GetStorageFormat(),
        mean->GetOriginalFormat());
    auto batchInvstd = executor->AllocTensor(
        variance->GetStorageShape(), variance->GetOriginalShape(), variance->GetDataType(),
        variance->GetStorageFormat(), variance->GetOriginalFormat());
    auto meanFinal = executor->AllocTensor(
        mean->GetStorageShape(), mean->GetOriginalShape(), mean->GetDataType(), mean->GetStorageFormat(),
        mean->GetOriginalFormat());
    auto batchInvstdFinal = executor->AllocTensor(
        variance->GetStorageShape(), variance->GetOriginalShape(), variance->GetDataType(),
        variance->GetStorageFormat(), variance->GetOriginalFormat());

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        SyncBatchNormGatherStats, OP_INPUT(totalSum, totalSquareSum, sampleCount, mean, variance),
        OP_OUTPUT(batchMean, batchInvstd, meanFinal, batchInvstdFinal), OP_ATTR(momentum, eps));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SyncBatchNormGatherStats ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr};
    }
    return {batchMean, batchInvstd, meanFinal, batchInvstdFinal};
}
} // namespace l0op