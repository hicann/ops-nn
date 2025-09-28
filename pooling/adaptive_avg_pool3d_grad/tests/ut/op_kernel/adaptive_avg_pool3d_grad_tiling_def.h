/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ADAPTIVE_AVG_POOL3D_GRAD_TILING_DEF_H__
#define __ADAPTIVE_AVG_POOL3D_GRAD_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AdaptiveAvgPool3dGradTilingData {
    uint64_t ncNum = 0;
    uint64_t dIn = 0;
    uint64_t hIn = 0;
    uint64_t wIn = 0;

    uint64_t dOut = 0;
    uint64_t hOut = 0;
    uint64_t wOut = 0;

    uint64_t taskCoreUsed = 0;
    uint64_t taskNumPerCore = 0;
    uint64_t taskNumLastCore = 0;
    uint64_t yNumPerCalc = 0;

    uint64_t ncSliceNum = 0;
    uint64_t ncAlignSliceLength = 0;
    uint64_t ncAlignSliceTail = 0;

    uint64_t isAtomicAdd = 0;
    uint64_t deterministicFlag = 0;
};
#pragma pack()

inline void InitAdaptiveAvgPool3dGradTilingData(uint8_t* tiling, AdaptiveAvgPool3dGradTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveAvgPool3dGradTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AdaptiveAvgPool3dGradTilingData tiling_data; \
    InitAdaptiveAvgPool3dGradTilingData(tiling_arg, &tiling_data)

#endif // __ADAPTIVE_AVG_POOL3D_GRAD_TILING_DEF_H__
