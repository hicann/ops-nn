/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef __ADAPTIVE_MAX_POOL3D_TILING_DEF_H__
#define __ADAPTIVE_MAX_POOL3D_TILING_DEF_H__
#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AdaptiveMaxPool3dTilingData {
    uint64_t N{0};
    uint64_t C{0};
    uint64_t Di{0};
    uint64_t Hi{0};
    uint64_t Wi{0};
    uint64_t Do{0};
    uint64_t Ho{0};
    uint64_t Wo{0};
    uint64_t coreNums{0};
    uint64_t useCoreNum{0};
    uint64_t totalIdx{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t doFactor{0};
    uint64_t hoFactor{0};
    uint64_t woFactor{0};
    uint64_t ncOuter{0};
    uint64_t doOuter{0};
    uint64_t hoOuter{0};
    uint64_t woOuter{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
};
#pragma pack()

#pragma pack(1)
struct AdaptiveMaxPool3dSmallPoolTilingData {
    uint64_t N{0};
    uint64_t C{0};
    uint64_t Di{0};
    uint64_t Hi{0};
    uint64_t Wi{0};
    uint64_t Do{0};
    uint64_t Ho{0};
    uint64_t Wo{0};
    uint64_t coreNums{0};
    uint64_t useCoreNum{0};
    uint64_t totalIdx{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t doFactor{0};
    uint64_t hoFactor{0};
    uint64_t woFactor{0};
    uint64_t ncOuter{0};
    uint64_t doOuter{0};
    uint64_t hoOuter{0};
    uint64_t woOuter{0};
    uint64_t ncTail{0};
    uint64_t doTail{0};
    uint64_t hoTail{0};
    uint64_t woTail{0};
};
#pragma pack()

#pragma pack(1)
struct AdaptiveMaxPool3dBigPoolTilingData {
    uint64_t N{0};
    uint64_t C{0};
    uint64_t Di{0};
    uint64_t Hi{0};
    uint64_t Wi{0};
    uint64_t Do{0};
    uint64_t Ho{0};
    uint64_t Wo{0};
    uint64_t coreNums{0};
    uint64_t useCoreNum{0};
    uint64_t totalIdx{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
};
#pragma pack()

inline void InitTilingData(uint8_t* tiling, AdaptiveMaxPool3dTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveMaxPool3dTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

inline void InitTilingData(uint8_t* tiling, AdaptiveMaxPool3dSmallPoolTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveMaxPool3dSmallPoolTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

inline void InitTilingData(uint8_t* tiling, AdaptiveMaxPool3dBigPoolTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveMaxPool3dBigPoolTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg)      \
    AdaptiveMaxPool3dSmallPoolTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif // __ADAPTIVE_MAX_POOL3D_TILING_DEF_H__