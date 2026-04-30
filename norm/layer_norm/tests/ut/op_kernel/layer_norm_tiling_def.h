/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_tiling_def.h
 * \brief
 */

#ifndef __LAYER_NORM_TILING_DEF_H__
#define __LAYER_NORM_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X float_t
#define DTYPE_GAMMA float_t
#define DTYPE_MEAN float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct LayerNormV3TilingDataRegBaseTwoPass {
    int64_t r;
    int64_t rAlign;
    int64_t a;
    int64_t aFactor;
    int64_t aBlockFactor;
    int64_t blockNum;
    int64_t binaryAddQuotient;
    int64_t binaryAddK;
    int64_t binaryAddLastNum;
    int64_t powerOfTwoForR;
    int64_t tmpBufferSize;
    int64_t nullptrGamma;
    int64_t nullptrBeta;
    float epsilon;
    LayerNormSeparateTiling layerNormTiling;
};

struct LayerNormV3TilingDataWelford {
    int64_t M;
    int64_t N;
    int64_t rAlign;
    int64_t numBlocks;
    int64_t mainBlockCount;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;
    int64_t tileLength;
    int64_t welfordTempSize;
    int64_t welfordUpdateTimes;
    int64_t welfordUpdateTail;
    int64_t nullptrGamma;
    int64_t nullptrBeta;
    int64_t apiTempBufferSize;
    float epsilon;
};

struct LayerNormV3TilingDataRegBaseTwoPassPerf {
    int64_t a;
    int64_t aBlockFactor;
    int32_t aUbFactor;
    int32_t aUbFactorAlignB32;
    int32_t r;
    int32_t rAlign;
    int32_t formerBlockUbLoops;
    int32_t tailBlockUbLoops;
    int32_t powerOfTwoForR;
    float epsilon;
    int8_t nullptrGamma;
    int8_t nullptrBeta;
};

struct LayerNormV3TilingDataRegBaseNoReduce {
    int64_t a;
    int64_t aBlockFactor;
    int32_t aUbFactor;
    int32_t aUbFactorAlignB32;
    int32_t formerBlockUbLoops;
    int32_t tailBlockUbLoops;
    float epsilon;
    int8_t nullptrGamma;
    int8_t nullptrBeta;
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