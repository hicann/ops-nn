/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _ADA_LAYER_NORM_GRAD_TILING_H_
#define _ADA_LAYER_NORM_GRAD_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)
struct AdaLayerNormGradTilingDataCommon {
int64_t batch = 0;
int64_t seq = 0;
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
int64_t blockFormerScaleBufferBytes = 0;
int64_t blockTailScaleBufferBytes = 0;
int64_t wholeBufferElemNums = 0;
int64_t blockFormerScaleBufferElemNums = 0;
int64_t blockTailScaleBufferElemNums = 0;
};

#pragma pack()

#pragma pack(1)
struct AdaLayerNormGradTilingDataMergeBSCommon {
int64_t row = 0;
int64_t col = 0;
int64_t batch = 0;
int64_t seq = 0;
int64_t colAlignM = 0;
int64_t colAlignV = 0;
int64_t blockNum = 0;
int64_t blockFormer = 0;
int64_t blockTail = 0;
int64_t ubFormer = 0;
int64_t blockFormerScaleBufferBytes = 0;
int64_t blockTailScaleBufferBytes = 0;
int64_t wholeBufferElemNums = 0;
int64_t blockFormerScaleBufferElemNums = 0;
int64_t blockTailScaleBufferElemNums = 0;
int64_t ubLoopOfFormerBlock = 0;
int64_t ubLoopOfTailBlock = 0;
int64_t ubTailOfFormerBlock = 0;
int64_t ubTailOfTailBlock = 0;
int64_t wholeBufferBytes = 0;
int64_t lastRBufferBytes = 0;
int64_t nlastRBufferBytes = 0;
int64_t lastBrcbBufferBytes = 0;
};

#pragma pack()

#pragma pack(1)
struct AdaLayerNormGradTilingDataWorkspace {
int64_t batch = 0;
int64_t seq = 0;
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
struct AdaLayerNormGradTilingDataMergeBSWorkspace {
int64_t row = 0;
int64_t col = 0;
int64_t batch = 0;
int64_t seq = 0;
int64_t colAlignM = 0;
int64_t colAlignV = 0;
int64_t blockNum = 0;
int64_t blockFormer = 0;
int64_t blockTail = 0;
int64_t ubLoop = 0;
int64_t ubFormer = 0;
int64_t ubTail = 0;
};

#pragma pack()



#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AdaLayerNormGradTilingDataCommon* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(AdaLayerNormGradTilingDataCommon) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AdaLayerNormGradTilingDataCommon* const_data)
{
    memcpy(const_data, tiling, sizeof(AdaLayerNormGradTilingDataCommon));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AdaLayerNormGradTilingDataMergeBSCommon* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(AdaLayerNormGradTilingDataMergeBSCommon) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AdaLayerNormGradTilingDataMergeBSCommon* const_data)
{
    memcpy(const_data, tiling, sizeof(AdaLayerNormGradTilingDataMergeBSCommon));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AdaLayerNormGradTilingDataWorkspace* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(AdaLayerNormGradTilingDataWorkspace) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AdaLayerNormGradTilingDataWorkspace* const_data)
{
    memcpy(const_data, tiling, sizeof(AdaLayerNormGradTilingDataWorkspace));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, AdaLayerNormGradTilingDataMergeBSWorkspace* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(AdaLayerNormGradTilingDataMergeBSWorkspace) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, AdaLayerNormGradTilingDataMergeBSWorkspace* const_data)
{
    memcpy(const_data, tiling, sizeof(AdaLayerNormGradTilingDataMergeBSWorkspace));
}
#endif


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif