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
 * \file lp_norm_v2_tiling_def.h
 * \brief
 */

#ifndef __LP_NORM_V2_TILING_DEF_H__
#define __LP_NORM_V2_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X float_t
#define DTYPE_Y float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct ReduceOpTilingDataDef {
    uint64_t factorACntPerCore;
    uint64_t factorATotalCnt;
    uint64_t ubFactorA;
    uint64_t factorRCntPerCore;
    uint64_t factorRTotalCnt;
    uint64_t ubFactorR;
    uint64_t groupR;
    uint64_t outSize;
    uint64_t basicBlock;
    uint64_t resultBlock;
    int32_t coreNum;
    int32_t useNddma;
    float meanVar;
    uint64_t shape[8];
    int64_t stride[8];
    int64_t dstStride[8];
    uint64_t sliceNum[8];
    uint64_t sliceShape[8];
    uint64_t sliceStride[8];
};

struct LpNormV2TilingData {
    ReduceOpTilingDataDef reduceTiling;
    float epsilon;
    float p;
    float recp;
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