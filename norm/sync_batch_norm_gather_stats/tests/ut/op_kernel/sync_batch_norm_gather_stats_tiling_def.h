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
 * \file sync_batch_norm_gather_stats_tiling_def.h
 * \brief
 */

#ifndef __SYNC_BATCH_NORM_GATHER_STATS_TILING_DEF_H__
#define __SYNC_BATCH_NORM_GATHER_STATS_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_fp16.h"

using float16_t = half;

#define DTYPE_TOTAL_SUM float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct SyncBatchNormGatherStatsTilingData {
    uint64_t blockDim;
    uint64_t blockFormer;
    uint64_t blockTail;
    uint64_t nLen;
    uint64_t cLen;
    uint64_t ubFormer;
    uint64_t ubTail;
    float momentum;
    float eps;
};

struct SyncBatchNormGatherStatsNNotFullLoadTilingData {
    int64_t blockDim;
    int64_t cLen;
    int64_t cFactor;
    int64_t cLoopMainBlock;
    int64_t cTileMainBlock;
    int64_t cLoopTailBlock;
    int64_t cTailTailBlock;
    int64_t nFactor;
    int64_t nLoop;
    int64_t nMainFoldCount;
    int64_t nTail;
    int64_t cacheBufferCount;
    int32_t resultCacheId;
    float momentum;
    float eps;
};
#pragma pack()

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#endif