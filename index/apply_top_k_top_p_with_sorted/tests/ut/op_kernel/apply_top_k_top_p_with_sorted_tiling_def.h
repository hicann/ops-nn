/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef APPLY_TOP_K_TOP_P_WITH_SORTED_TILING_H_TEST_KERNEL
#define APPLY_TOP_K_TOP_P_WITH_SORTED_TILING_H_TEST_KERNEL

#include "kernel_tiling/kernel_tiling.h"

struct ApplyTopKTopPWithSortedTilingData {
    uint64_t batchSize = 0;
    uint64_t vocabSize = 0;
    uint64_t batchPerCore = 0;
    uint64_t tailBatch = 0;
    uint64_t blockNum = 0;
    uint64_t dataNumInit = 0;
    uint64_t dataNumInitAligned = 0;
    uint64_t ubFactorElement = 0;
    uint64_t ubFactorElementAligned = 0;
    uint64_t tailUbFactorElement = 0;
    uint64_t tailUbFactorElementAligned = 0;
    uint64_t calUbSize = 0;
    uint64_t iterateTimes = 0;
};

inline void InitApplyTopKTopPWithSortedTilingData(uint8_t* tiling, ApplyTopKTopPWithSortedTilingData* data)
{
    memcpy(data, tiling, sizeof(ApplyTopKTopPWithSortedTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    ApplyTopKTopPWithSortedTilingData tiling_data;      \
    InitApplyTopKTopPWithSortedTilingData(tiling_arg, &tiling_data)
#endif // APPLY_TOP_K_TOP_P_WITH_SORTED_TILING_H_TEST_KERNEL
