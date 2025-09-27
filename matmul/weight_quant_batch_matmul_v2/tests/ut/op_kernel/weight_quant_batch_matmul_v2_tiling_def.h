/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H__
#define __WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#else
#define __aicore__ [aicore]
#endif
#pragma pack(1)
struct WeightQuantBatchMatmulV2MsdTilingData {
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimM = 0;
    uint8_t cubeBlockDimK = 0;
    uint8_t hasBias = 0;
    uint16_t v1BaseM = 0;
    uint16_t preloadTimes = 0;
    uint16_t taskNSize = 0;
    uint16_t taskSingleCoreNSize = 0;
    uint16_t postProcessBaseM = 0;
    uint16_t postProcessSingleCoreM = 0;
    uint32_t preProcessUsedVecNum = 0;
    uint32_t v1BaseK = 0;
    uint64_t mSize = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupPack = 0;
    uint64_t groupSize = 0;
    TCubeTiling matmulTiling;
};
#pragma pack()

#pragma pack(1)
struct WeightQuantBatchMatmulV2MsdGroupTilingData {
    uint8_t vecBlockDimN = 0;
    uint8_t cubeBlockDimK = 0;
    uint8_t cubeBlockDimN = 0;
    uint8_t vec1SingleCoreM = 0;
    uint8_t hasBias = 0;
    uint8_t reserve1 = 0;
    uint16_t reserve2 = 0;
    uint32_t reserve3 = 0;
    uint32_t singleCoreK = 0;
    uint32_t vecSingleCoreN = 0;
    uint32_t singleCoreGroup = 0;
    uint64_t mSize = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupSize = 0;
    TCubeTiling matmulTiling;
};
#pragma pack()

#pragma pack(1)
struct WeightQuantBatchMatmulV2TilingData {
    uint8_t vecBlockDimN = 0;
    uint8_t vecBlockDimK = 0;
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimM = 0;
    uint8_t cubeBlockDimK = 0;
    uint8_t kPadSize = 0;
    uint8_t nPadSize = 0;
    uint8_t haveBatchA = 0;
    uint8_t haveBatchB = 0;
    uint8_t reserve1 = 0;
    uint8_t reserve2 = 0;
    uint8_t reserve3 = 0;
    uint16_t vecSingleKGroupNum = 0;
    uint16_t vecSingleKTailGroupNum = 0;
    uint16_t AL1Pingpong = 0;
    uint16_t BL1Pingpong = 0;
    uint32_t vecSingleK = 0;
    uint32_t vecSingleN = 0;
    uint32_t vec2SingleM = 0;
    uint32_t vecSingleKTail = 0;
    uint32_t vecSingleNTail = 0;
    uint32_t wInQueueSize = 0;
    uint32_t offsetInQueueSize = 0;
    uint32_t scaleInQueueSize = 0;
    uint32_t wOutQueueSize = 0;
    uint32_t antiQuantTmpBufferSize = 0;
    uint32_t vecCubeNRatio = 0;
    uint32_t vecCubeTailNRatio = 0;
    uint32_t vecCubeKRatio = 0;
    uint32_t vecCubeTailKRatio = 0;
    uint32_t cubeTailM = 0;
    uint32_t cubeTailN = 0;
    uint32_t cubeSingleNLoop = 0;
    uint32_t cubeSingleNTailLoop = 0;
    uint32_t repeatAxisMax = 0;
    uint64_t vecSingleKLoop = 0;
    uint64_t vecSingleNLoop = 0;
    uint64_t vecSingleKTailLoop = 0;
    uint64_t vecSingleNTailLoop = 0;
    uint64_t vec2SingleMLoop = 0;
    uint64_t kAlign = 0;
    uint64_t nAlign = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t groupSize = 0;
    uint64_t mSize = 0;
    uint64_t blockBatch = 0;
    uint64_t shapeBatch = 0;
    uint64_t mAubSize = 0;
    uint64_t kAubSize = 0;
    uint64_t nBubSize = 0;
    uint64_t kBubSize = 0;
    uint64_t mCubSize = 0;
    uint64_t nCubSize = 0;
    uint64_t mAL1Size = 0;
    uint64_t kAL1Size = 0;
    uint64_t nBL1Size = 0;
    uint64_t kBL1Size = 0;
    TCubeTiling matmulTiling;
};
#pragma pack()

#pragma pack(1)
struct WeightQuantBatchMatmulV2CustomNzSplitKTilingData {
    uint8_t hasBias = 0;
    uint8_t reverse1 = 0;
    uint16_t reverse2 = 0;
    uint8_t vecBlockDimN = 0;
    uint8_t vecBlockDimK = 0;
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimK = 0;

    uint16_t postSingleN = 0;
    uint16_t postSingleM = 0;

    uint32_t vecSingleN = 0;
    uint32_t singleK = 0;
    uint32_t cubeSingleM = 0;
    uint32_t cubeSingleN = 0;
    uint32_t cubeBaseK = 0;
    uint64_t postSingleCoreN = 0;

    uint64_t cubeSingleCoreNLoop = 0;
    uint64_t cubeSingleCoreNTailLoop = 0;
    uint64_t singleCoreKLoop = 0;
    uint64_t singleCoreKTailLoop = 0;
    uint64_t vecSingleCoreNLoop = 0;
    uint64_t vecSingleCoreNTailLoop = 0;

    uint64_t cubeSingleCoreN = 0;
    uint64_t cubeSingleCoreNTail = 0;
    uint64_t cubeSingleCoreNOriTail = 0;
    uint64_t singleCoreK = 0;
    uint64_t singleCoreKTail = 0;
    uint64_t singleCoreKOriTail = 0;
    uint64_t vectorSingleCoreN = 0;
    uint64_t vectorSingleCoreNTail = 0;

    uint64_t mSize = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t nSizeAlign = 0;
    uint64_t kSizeAlign = 0;
};
#pragma pack()

#if defined(__CCE_KT_TEST__)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t* p_tilingdata, T* tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(
    const __gm__ uint8_t* p_tilingdata, T* tilingdata)
#endif
{
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_CUBE__)
    uint64_t all_bytes = sizeof(T);
    uint64_t res_bytes = all_bytes % 8;
    uint64_t i = 0;
    for (; i + 7 < all_bytes; i += 8) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p_tilingdata + i));
    }
    if (res_bytes & 0b100) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint32_t*)((const __gm__ uint8_t*)p_tilingdata + i));
        i += 4;
    }
    if (res_bytes & 0b10) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint16_t*)((const __gm__ uint8_t*)p_tilingdata + i));
        i += 2;
    }
    if (res_bytes & 0b1) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint8_t*)((const __gm__ uint8_t*)p_tilingdata + i));
    }
#else
    LocalTensor<uint8_t> tilingDataInUb;
    GlobalTensor<uint8_t> tilingDataInGm;
    tilingDataInGm.SetGlobalBuffer((__gm__ uint8_t *)p_tilingdata);
    tilingDataInUb.InitBuffer(0, 192 * 1024); // 192 * 1024是UB大小
    tilingDataInUb.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    __ubuf__ uint8_t* tilingdata_in_ub = (__ubuf__ uint8_t *)tilingDataInUb.GetPhyAddr();
    uint32_t len_burst = (sizeof(T) + 31) / 32;
    DataCopy(tilingDataInUb, tilingDataInGm, {1, len_burst, 0, 0});
    PipeBarrier<PIPE_ALL>();
    uint64_t all_bytes = sizeof(T);
    uint64_t res_bytes = all_bytes % 8;
    uint64_t i = 0;
    for (; i + 7 < all_bytes; i += 8) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
    }
    if (res_bytes & 0b100) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint32_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
        i += 4;
    }
    if (res_bytes & 0b10) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint16_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
        i += 2;
    }
    if (res_bytes & 0b1) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint8_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
    }
    PipeBarrier<PIPE_ALL>();
#endif
}
#define GET_TILING_DATA(tiling_data, tiling_arg)    \
    WeightQuantBatchMatmulV2TilingData tiling_data; \
    InitTilingData<WeightQuantBatchMatmulV2TilingData>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto typeVar##var = ((tiling_type*)0)->member;               \
    decltype(typeVar##var) var;                                  \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);   \
    InitTilingData<decltype(typeVar##var)>(tiling + offset##var, &var);

#endif