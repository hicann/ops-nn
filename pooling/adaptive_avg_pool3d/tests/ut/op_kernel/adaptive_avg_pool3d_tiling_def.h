/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __ADAPTIVE_AVG_POOL3D_TILING_DEF_H__
#define __ADAPTIVE_AVG_POOL3D_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AdaptiveAvgPool3dTilingData {
    uint64_t dimC = 0;
    uint64_t cTileLength = 0;
    uint64_t inD = 0;
    uint64_t inH = 0;
    uint64_t inW = 0;
    uint64_t outD = 0;
    uint64_t outH = 0;
    uint64_t outW = 0;
    uint64_t formerLength = 0;
    uint64_t formerNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailNum = 0;
    uint64_t indexBufLen = 0;
    uint64_t windowWNum = 0;
    uint64_t maxWindowWLength = 0;
    uint64_t inputTileNum = 0;
    uint64_t atomicAddNum = 0;
};
#pragma pack()

inline void InitAdaptiveAvgPool3dTilingData(uint8_t* tiling, AdaptiveAvgPool3dTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveAvgPool3dTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AdaptiveAvgPool3dTilingData tiling_data;     \
    InitAdaptiveAvgPool3dTilingData(tiling_arg, &tiling_data)

#endif // __ADAPTIVE_AVG_POOL3D_TILING_DEF_H__