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
 * \file fused_linear_online_max_sum_tiling_data.h
 * \brief
 */
#pragma once

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

namespace FusedLinearOnlineMaxSum {
constexpr uint64_t STRUCT_ALIGNAS = 8;
#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS) FusedLinearOnlineMaxSumTilingData {
    uint64_t m;
    uint64_t k;
    uint64_t n;
    uint64_t bufSize;
    uint64_t cubeCoreNum;
    uint64_t vecCoreNum;
    uint64_t batchTaksPerVecCore;
    uint64_t batchTaksTailVecCore;
    uint64_t targetTasksPerLoop;
    float vocabStartIndex;
    float vocabEndIndex;
    uint64_t initWorkspaceLength;
    uint64_t cubeCoreNumAligned;
    uint64_t matmulInputEmptyFlag;
    TCubeTiling mmTiling;
};
#pragma pack(pop)
} // namespace FusedLinearOnlineMaxSum
