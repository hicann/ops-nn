/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __quant_batch_matmul_v4_TILING_H__
#define __quant_batch_matmul_v4_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#else
#define __aicore__ [aicore]
#endif
#pragma pack(1)
struct QuantBatchMatmulV4TilingData {
    uint8_t vecBlockDimN = 0;
    uint8_t vecBlockDimK = 0;
    uint8_t cubeBlockDimN = 0;
    uint8_t cubeBlockDimM = 0;
    uint8_t cubeBlockDimK = 0;
    uint8_t kPadSize = 0;
    uint8_t nPadSize = 0;
    uint8_t haveBatchA = 0;
    uint8_t haveBatchB = 0;
    uint8_t vecCoreParallel = 0;
    uint8_t reserve1 = 0;
    uint8_t reserve2 = 0;
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
    bool hasX1Scale = false;
    bool hasX2Scale = false;
    TCubeTiling matmulTiling;
}__attribute__((__may_alias__));
#pragma pack()

#pragma pack(1)
struct QuantBatchMatmulV4MsdTilingData {
    uint8_t coreNum = 0;
    uint32_t vBaseM = 0;
    uint32_t ubRestBytes = 0;
    uint32_t parallNum = 0;
    uint32_t ubCalSize = 0;
    uint32_t mSize = 0;
    uint32_t kSize = 0;
    uint32_t nSize = 0;
    uint32_t groupSize = 0;
    TCubeTiling matmulTiling;
};
#pragma pack()

#pragma pack(1)
struct L2cacheTileParam {
    uint32_t mTileCntL2 = 0;
    uint32_t nTileCntL2 = 0;
    uint32_t mTileBlock = 0;
    uint32_t nTileBlock = 0;
    uint32_t calOrder = 0;
    uint32_t isBasicTiling = 0;
};
#pragma pack()

#pragma pack(1)
struct QuantBatchMatmulV4PerblockTilingData {
    uint32_t groupSizeM = 0;
    uint32_t groupSizeN = 0;
    uint32_t groupSizeK = 0;
    uint32_t ubCalcM = 0;
    uint32_t ubCalcN = 0;
    bool transA = false;
    bool transB = true;
    TCubeTiling matmulTiling;
    L2cacheTileParam tileL2cacheTiling;
};
#pragma pack()

#if defined(__CCE_KT_TEST__)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#endif
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__GET_CODE_CHANNEL__)
#if defined(__DAV_C100__) || defined(__CCE_KT_TEST__)
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i));
            (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint32_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint16_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint8_t *)((const __gm__ uint8_t *)p_tilingdata + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__gm__ uint8_t *)p_tilingdata, all_bytes);
#endif
#else
    LocalTensor<uint8_t> tilingDataInUb;
    GlobalTensor<uint8_t> tilingDataInGm;
    tilingDataInGm.SetGlobalBuffer((__gm__ uint8_t *)p_tilingdata);
    tilingDataInUb.InitBuffer(0, 192 * 1024); // 192 * 1024是UB大小
    tilingDataInUb.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)tilingDataInUb.GetPhyAddr();
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
#if defined(__DAV_C310__)
    DataCopyPad(tilingDataInUb, tilingDataInGm, {1, len_burst * 32, 0, 0}, {false, 0, 0, 0});
#else
    DataCopy(tilingDataInUb, tilingDataInGm, {1, len_burst, 0, 0});
#endif
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
#if defined(__DAV_C100__)
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
            (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t *)tilingdata_in_ub + i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint16_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint8_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__ubuf__ uint8_t *)tilingdata_in_ub, all_bytes);
#endif
#endif
#ifndef __CCE_KT_TEST__
    PipeBarrier<PIPE_ALL>();
#endif
}
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    QuantBatchMatmulV4TilingData tiling_data; \
    InitTilingData<QuantBatchMatmulV4TilingData>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto typeVar##var = ((tiling_type *)0)->member;                   \
    decltype(typeVar##var) var;                                       \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);        \
    InitTilingData<decltype(typeVar##var)>(tiling + offset##var, &var);


#endif