/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SYNC_BATCH_NORM_GATHER_STATS_FUSED_TILING_DEF_H_
#define _SYNC_BATCH_NORM_GATHER_STATS_FUSED_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)
struct SyncBatchNormGatherStatsFusedTilingDataCommon {
    int64_t nLength = 0;
    int64_t cLength = 0;
    int64_t cFormerAlignV_ = 0;
    int64_t cFormerAlignM_ = 0;
    int64_t cTailAlignV_ = 0;
    int64_t cTailAlignM_ = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoop = 0;
    int64_t ubTail = 0;
    int64_t wholeBufferByteSize = 0;
    int64_t nBufferByteSize = 0;
    int64_t cBufferByteSize = 0;
    int64_t nBrcbBufferByteSize = 0;
    int64_t wholeBufferElemNums = 0;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
#pragma pack()

#pragma pack(1)
struct SyncBatchNormGatherStatsFusedTilingDataWorkspace {
    int64_t nLength = 0;
    int64_t cLength = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFormerOfFormer = 0;
    int64_t ubTailOfFormer = 0;
    int64_t ubLoopOfFormer = 0;
    int64_t ubFormerOfTail = 0;
    int64_t ubTailOfTail = 0;
    int64_t ubLoopOfTail = 0;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
#pragma pack()

#pragma pack(1)
struct SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon {
    int64_t nLength = 0;
    int64_t cLength = 0;
    int64_t cAlignV = 0;
    int64_t cAlignM = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    int64_t wholeBufferByteSize = 0;
    int64_t nBufferByteSize = 0;
    int64_t cBufferByteSize = 0;
    int64_t nBrcbBufferByteSize = 0;
    int64_t wholeBufferElemNums = 0;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
#pragma pack()

#pragma pack(1)
struct SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace {
    int64_t nLength = 0;
    int64_t cLength = 0;
    int64_t cAlignV = 0;
    int64_t blockFormer = 0;
    int64_t blockNum = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubTail = 0;
    int64_t ubLoop = 0;
    float momentum = 0.1f;
    float eps = 1e-5f;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataCommon* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SyncBatchNormGatherStatsFusedTilingDataCommon) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataCommon* const_data)
{
    memcpy(const_data, tiling, sizeof(SyncBatchNormGatherStatsFusedTilingDataCommon));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataWorkspace* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SyncBatchNormGatherStatsFusedTilingDataWorkspace) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataWorkspace* const_data)
{
    memcpy(const_data, tiling, sizeof(SyncBatchNormGatherStatsFusedTilingDataWorkspace));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon* const_data)
{
    memcpy(const_data, tiling, sizeof(SyncBatchNormGatherStatsFusedTilingDataFirstAxisCommon));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace* const_data)
{
    memcpy(const_data, tiling, sizeof(SyncBatchNormGatherStatsFusedTilingDataFirstAxisWorkspace));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif