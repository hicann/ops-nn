/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _LAYER_NORM_GRAD_V3_TILING_H_
#define _LAYER_NORM_GRAD_V3_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct LayerNormGradV3TilingDataSingleRead {
    int64_t row = 0;
    int64_t col = 0;
    int64_t colAlignM = 0;
    int64_t colAlignV = 0;
    int64_t blockNum = 0;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    int64_t bufferElemNums = 0;
};

#pragma pack()

#pragma pack(1)

struct LayerNormGradV3TilingDataTranspose {
    uint64_t row = 0;                         // 输入tensor的行
    uint64_t col = 0;                         // 输入tensor的列，即reduce的轴
    uint64_t numBlocks = 0;                    // 实际使用的core数量
    uint64_t blockFormer = 0;                 // 整核处理的row大小
    uint64_t blockTail = 0;                   // 尾核处理的row大小
    uint64_t ubFormer = 0;                    // ub整循环处理的row大小
    uint64_t ubLoopOfFormerBlock = 0;         // 整核处理的ub循环次数
    uint64_t ubLoopOfTailBlock = 0;           // 尾核处理的ub循环次数
    uint64_t ubTailOfFormerBlock = 0;         // 整核ub尾循环处理的row大小
    uint64_t ubTailOfTailBlock = 0;           // 尾核ub尾循环处理的row大小
    uint64_t bFormer = 0;                     // ubFormer借轴大小，ubFormer->16*bFormer
    uint64_t dichotomizeAddDiffSize = 0;      // row与小于row的最近二次幂的差值
    uint64_t deterministicComputeWspSize = 0; // 确定性计算需要的pdGamma或pdBeta workspace size大小
    float coefficient = 0;                    // 1/col
    float placeHolder = 0;
};

#pragma pack()

#pragma pack(1)
struct LayerNormGradV3TilingDataCommon {
    int64_t row = 0;
    int64_t col = 0;
    int64_t colAlignM = 0;
    int64_t colAlignV = 0;
    int64_t blockNum = 0;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    int64_t ubFormer = 0;
    int64_t ubLoopOfFormerBlock = 0;
    int64_t ubLoopOfTailBlock = 0;
    int64_t ubTailOfFormerBlock = 0;
    int64_t ubTailOfTailBlock = 0;
    int64_t wholeBufferBytes = 0;
    int64_t lastRBufferBytes = 0;
    int64_t nlastRBufferBytes = 0;
    int64_t lastBrcbBufferBytes = 0;
    int64_t wholeBufferElemNums = 0;
};

#pragma pack()

#pragma pack(1)
struct LayerNormGradV3TilingDataWorkspace {
    int64_t row = 0;
    int64_t col = 0;
    int64_t blockNum = 0;
    int64_t blockFormer = 0;
    int64_t blockTail = 0;
    int64_t ubLoop = 0;
    int64_t ubFormer = 0;
    int64_t ubTail = 0;
    int64_t colAlignM = 0;
    int64_t colAlignV = 0;
};

#pragma pack()

#pragma pack(1)
struct LayerNormGradV3TilingDataRecompute {
    int64_t row = 0;
    int64_t col = 0;
    int64_t gammaBetaMfactor = 0;
    int64_t gammaBetaNfactor = 0;
    int64_t gammaBetaMainBlockFactor = 0;
    int64_t gammaBetanumBlocks = 0;
    int64_t gammaBetaMainBlockCount = 0;
    int64_t gammaBetaTailBlockCount = 0;
    int64_t gammaBetaTailBlockFactor = 0;
    int64_t gammaBetaMloop = 0;
    int64_t gammaBetaMTotalLoop = 0;
    int64_t gammaBetaMtail = 0;
    int64_t gammaBetaBasicBlockLoop = 0;
    int64_t gammaBetaMainFoldCount = 0;
    int64_t gammaBetaNfactorBlockAligned = 0;
    int64_t gammaBetaMfactorBlockAligned = 0;
    int64_t backwardMfactor = 0;
    int64_t backwardNfactor = 0;
    int64_t backwardMainBlockFactor = 0;
    int64_t backwardnumBlocks = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradV3TilingDataSingleRead* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradV3TilingDataSingleRead) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradV3TilingDataSingleRead* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradV3TilingDataSingleRead));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradV3TilingDataTranspose* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradV3TilingDataTranspose) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradV3TilingDataTranspose* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradV3TilingDataTranspose));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradV3TilingDataCommon* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradV3TilingDataCommon) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradV3TilingDataCommon* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradV3TilingDataCommon));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradV3TilingDataWorkspace* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradV3TilingDataWorkspace) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradV3TilingDataWorkspace* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradV3TilingDataWorkspace));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormGradV3TilingDataRecompute* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormGradV3TilingDataRecompute) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormGradV3TilingDataRecompute* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormGradV3TilingDataRecompute));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif