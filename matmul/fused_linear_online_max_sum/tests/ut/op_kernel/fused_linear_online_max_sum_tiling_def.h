/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL
#define FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL

#include "kernel_tiling/kernel_tiling.h"

struct FusedLinearOnlineMaxSumTilingData {
    uint64_t m = 0;
    uint64_t k = 0;
    uint64_t n = 0;
    uint64_t bufSize = 0;
    uint64_t cubeCoreNum = 0;
    uint64_t vecCoreNum = 0;
    uint64_t batchTaksPerVecCore = 0;
    uint64_t batchTaksTailVecCore = 0;
    uint64_t targetTasksPerLoop = 0;
    float vocabStartIndex = 0;
    float vocabEndIndex = 0;
    uint64_t initWorkspaceLength = 0;
    uint64_t cubeCoreNumAligned = 0;
    uint64_t matmulInputEmptyFlag = 0;
    TCubeTiling mmTiling;
};

inline void InitFusedLinearOnlineMaxSumTilingData(uint8_t* tiling, FusedLinearOnlineMaxSumTilingData* data)
{
    memcpy(data, tiling, sizeof(FusedLinearOnlineMaxSumTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)                    \
    FusedLinearOnlineMaxSumTilingData tiling_data;                  \
    InitFusedLinearOnlineMaxSumTilingData(tiling_arg, &tiling_data)
#endif // FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H_TEST_KERNEL
