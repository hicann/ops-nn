/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_v4_regbase_tiling.h
 * \brief
 */

#ifndef _LAYER_NORM_V4_REGBASE_TILING_H_
#define _LAYER_NORM_V4_REGBASE_TILING_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct LayerNormV4TilingDataRegBaseTwoPass {
    int64_t r = 0;
    int64_t rAlign = 0;
    int64_t a = 0;
    int64_t aFactor = 0;
    int64_t aBlockFactor = 0;
    int64_t blockNum = 0;
    int64_t binaryAddQuotient = 0;
    int64_t binaryAddK = 0;
    int64_t binaryAddLastNum = 0;
    int64_t powerOfTwoForR = 0;
    int64_t tmpBufferSize = 0;
    int64_t nullptrGamma = 0;
    int64_t nullptrBeta = 0;
    float epsilon = 0;
};

#pragma pack()

#pragma pack(1)

struct LayerNormV4TilingDataWelford {
    int64_t M = 0;
    int64_t N = 0;
    int64_t rAlign = 0;
    int64_t blockDim = 0;
    int64_t mainBlockCount = 0;
    int64_t mainBlockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t tileLength = 0;
    int64_t welfordUpdateTimes = 0;
    int64_t welfordUpdateTail = 0;
    int64_t nullptrGamma = 0;
    int64_t nullptrBeta = 0;
    float epsilon = 0;
    int64_t apiTempBufferSize = 0;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4TilingDataRegBaseTwoPass* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4TilingDataRegBaseTwoPass) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4TilingDataRegBaseTwoPass* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4TilingDataRegBaseTwoPass));
}
#endif

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, LayerNormV4TilingDataWelford* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(LayerNormV4TilingDataWelford) / 4; i++)
        *(dst + i) = *(src + i);
}
#else
inline void InitTilingData(uint8_t* tiling, LayerNormV4TilingDataWelford* const_data)
{
    memcpy(const_data, tiling, sizeof(LayerNormV4TilingDataWelford));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif