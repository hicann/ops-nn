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
 * \file sync_batch_norm_gather_stats.h
 * \brief
 */
#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_SYNC_BATH_NORM_GATHER_STATS_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_SYNC_BATH_NORM_GATHER_STATS_H_

#include "opdev/op_executor.h"

namespace l0op {
constexpr size_t SYNC_BATCH_NORM_OUTPUT_NUM = 4;
const std::array<aclTensor*, SYNC_BATCH_NORM_OUTPUT_NUM> SyncBatchNormGatherStats(
    const aclTensor* totalSum, const aclTensor* totalSquareSum, const aclTensor* sampleCount, aclTensor* mean,
    aclTensor* variance, float momentum, float eps, aclOpExecutor* executor);
} // namespace l0op

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_SYNC_BATH_NORM_GATHER_STATS_H_