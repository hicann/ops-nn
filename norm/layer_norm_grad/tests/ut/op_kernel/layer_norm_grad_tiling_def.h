/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _LAYER_NORM_GRAD_TILING_H_
#define _LAYER_NORM_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)
struct LayerNormGradTilingDataRecompute {
    int64_t row = 0;
    int64_t col = 0;

    int64_t gammaBetaMainBlockFactor = 0;
    int64_t gammaBetaNloopMainBlock = 0;
    int64_t gammaBetaNtailMainBlock = 0;
    int64_t gammaBetaNloopTailBlock = 0;
    int64_t gammaBetaNtailTailBlock = 0;
    int64_t gammaBetaMtail = 0;
    int64_t gammaBetaBasicBlockLoop = 0;
    int64_t gammaBetaMainFoldCount = 0;
    int64_t backwardMainBlockFactor = 0;
    int64_t backwardMainBlockCount = 0;
    int64_t backwardTailBlockCount = 0;
    int64_t backwardTailBlockFactor = 0;
    int64_t backwardMLoopMain = 0;
    int64_t backwardMLoopTail = 0;
    int64_t backwardMLoopTailTail = 0;
    int64_t backwardMTailTail = 0;
    int64_t backwardNLoopMain = 0;
    int64_t backwardNTotalLoopMain = 0;
    int64_t backwardNLoopTail = 0;
    int64_t backwardBasicBlockLoop = 0;
    int64_t backwardMainFoldCount = 0;
    int64_t backwardNfactorBlockAligned = 0;
    int64_t backwardMfactorBlockAligned = 0;
    int64_t backwardCeilVLCount = 0;
    int64_t backwardFoldPoint = 0;
    int64_t backwardFoldSize = 0;
    int32_t gammaBetaBlockDim = 0;
    int32_t gammaBetaCacheBufferCount = 0;
    int32_t gammaBetaResultCacheID = 0;
    int32_t gammaBetaNfactor = 0;
    int32_t gammaBetaMfactor = 0;
    int32_t backwardBlockDim = 0;
    int32_t backwardMfactor = 0;
    int32_t backwardNfactor = 0;
    int32_t backwardCacheBufferCountMain = 0;
    int32_t backwardResultCacheIDMain = 0;
    
    int32_t pdxIsRequire = 1;
    int32_t pdgammaIsRequire = 1;
    int32_t pdbetaIsRequire = 1;
    float epsilon = 0;
};

struct LayerNormGradTilingDataGroupedReduceBigM {
    int64_t row = 0;
    int64_t col = 0;
    
    int64_t gammaBetaUsableBlocks = 0;
    int64_t gammaBetaMPerBlock = 0;
    int64_t gammaBetaMReminder = 0;
    int64_t gammaBetaNloop = 0;
    int64_t gammaBetaNtail = 0;
    int64_t gammaBetaMfactorBlockAligned = 0;
    int64_t gammaBetaNfactorBlockAligned = 0;

    int64_t gammabetaMToProcessMainBlock = 0;
    int64_t gammabetaMLoopMainBlock = 0;
    int64_t gammabetaMTotalLoopMainBlock = 0;
    int64_t gammabetaMTailMainBlock = 0;
    int64_t gammabetaBasicBlockLoopMainBlock = 0;
    int64_t gammabetaMainFoldCountMainBlock = 0;
    int64_t gammabetaCacheBufferCountMainBlock = 0;
    int64_t gammabetaResultCacheIDMainBlock = 0;

    int64_t gammabetaMToProcessTailBlock = 0;
    int64_t gammabetaMLoopTailBlock = 0;
    int64_t gammabetaMTotalLoopTailBlock = 0;
    int64_t gammabetaMTailTailBlock = 0;
    int64_t gammabetaBasicBlockLoopTailBlock = 0;
    int64_t gammabetaMainFoldCountTailBlock = 0;
    int64_t gammabetaCacheBufferCountTailBlock = 0;
    int64_t gammabetaResultCacheIDTailBlock = 0;

    int64_t gammaBetaMTailStg2 = 0;
    int64_t gammaBetaMBasicBlockLoopStg2 = 0;
    int64_t gammaBetaMMainFoldCountStg2 = 0;
    int64_t gammaBetaMResultCacheIDStg2 = 0;

    int32_t pdxIsRequire = 0;
    int32_t pdgammaIsRequire = 0;
    int32_t pdbetaIsRequire = 0;
    float epsilon = 0;
};

struct LayerNormGradTilingDataGroupedReduceBigN {
    int64_t row = 0;
    int64_t col = 0;

    int64_t gammaBetaMainBlockFactor = 0;
    int64_t gammaBetaBlockDim = 0;
    int64_t gammaBetaNloopMainBlock = 0;
    int64_t gammaBetaNtailMainBlock = 0;
    int64_t gammaBetaNloopTailBlock = 0;
    int64_t gammaBetaNtailTailBlock = 0;
    int64_t gammaBetaMtail = 0;
    int64_t gammaBetaBasicBlockLoop = 0;
    int64_t gammaBetaMainFoldCount = 0;
    int64_t gammaBetaCacheBufferCount = 0;
    int64_t gammaBetaResultCacheID = 0;
    int64_t gammaBetaNfactor = 0;
    int64_t gammaBetaMfactor = 0;
    
    int64_t backwardBlockDim = 0;
    int64_t backwardNPerBlock = 0;
    int64_t backwardNRem = 0;
    int64_t nToProcessMain = 0;
    int64_t nToProcessTail = 0;
    int64_t backwardMTotalLoop = 0;
    int64_t backwardMtail = 0;
    int64_t backwardNloopMain = 0;
    int64_t backwardNtailMain = 0;
    int64_t backwardBasicBlockLoopMain = 0;
    int64_t backwardMainFoldCountMain = 0;
    int64_t backwardNfactorBlockAligned = 0;
    int64_t backwardMfactor = 0;
    int64_t backwardMfactorBlockAligned = 0;
    int64_t backwardCacheBufferCountMain = 0;
    int64_t backwardResultCacheIDMain = 0;
    int64_t backwardNloopTail = 0;
    int64_t backwardNtailTail = 0;
    int64_t backwardBasicBlockLoopTail = 0;
    int64_t backwardMainFoldCountTail = 0;
    int64_t backwardCacheBufferCountTail = 0;
    int64_t backwardResultCacheIDTail = 0;

    int32_t pdxIsRequire = 0;
    int32_t pdgammaIsRequire = 0;
    int32_t pdbetaIsRequire = 0;
    float epsilon = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradTilingDataRecompute* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradTilingDataRecompute) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradTilingDataRecompute* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradTilingDataRecompute));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradTilingDataGroupedReduceBigM* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradTilingDataGroupedReduceBigM) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradTilingDataGroupedReduceBigM* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradTilingDataGroupedReduceBigM));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradTilingDataGroupedReduceBigN* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradTilingDataGroupedReduceBigN) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradTilingDataGroupedReduceBigN* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradTilingDataGroupedReduceBigN));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif
