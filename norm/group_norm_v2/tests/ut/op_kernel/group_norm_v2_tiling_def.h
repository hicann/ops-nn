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
 * \file group_norm_v2_tiling_def.h
 * \brief
 */

#ifndef __GROUP_NORM_V2_TILING_DEF_H__
#define __GROUP_NORM_V2_TILING_DEF_H__

#include <cstdint>
#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DTYPE_X float_t
#define __CCE_UT_TEST__

#pragma pack(1)
struct GroupNormV2TilingData {
    int64_t numGroups;
    int64_t hwNum;
    int64_t elemNum;
    int64_t shapeC;
    int64_t shapeD;
    int64_t realCoreNum;
    int64_t numPerCore;
    int64_t numLastCore;
    int64_t processSize;
    int64_t loopNum;
    int64_t loopTail;
    int64_t innerLoopNum;
    int64_t innerLoopTail;
    int64_t tilingKey;
    float epsilon;
    int64_t parallelN;
    int64_t ubSize;
    int64_t dichotomyAddPower;
    int64_t dichotomyAddK;
    int64_t dichotomyAddLastNum;
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

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    GET_TILING_DATA_WITH_STRUCT(GroupNormV2TilingData, tiling_data, tiling_arg)

#endif