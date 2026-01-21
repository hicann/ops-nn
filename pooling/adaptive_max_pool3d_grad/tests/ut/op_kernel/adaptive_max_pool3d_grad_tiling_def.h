/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _ADA_MAX_POOL3d_GRAD_TILING_DEF_H_
#define _ADA_MAX_POOL3d_GRAD_TILING_DEF_H_

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AdaptiveMaxPool3DGradTilingData {
    uint64_t ncDim = 0;
    uint64_t diDim = 0;
    uint64_t hiDim = 0;
    uint64_t wiDim = 0;
    uint64_t doDim = 0;
    uint64_t hoDim = 0;
    uint64_t woDim = 0;
    uint64_t kdMax = 0;
    uint64_t khMax = 0;
    uint64_t kwMax = 0;
    uint64_t baseNc = 0;
    uint64_t baseDo = 0;
    uint64_t baseHo = 0;
    uint64_t baseWo = 0;
    uint64_t singleCoreNc = 0;
    uint64_t singleCoreDo = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    uint64_t ncTail = 0;
    uint64_t doTail = 0;
    uint64_t hoTail = 0;
    uint64_t woTail = 0;
    uint64_t ncCnt = 0;
    uint64_t doCnt = 0;
    uint64_t hoCnt = 0;
    uint64_t woCnt = 0;
    uint64_t totalCnt = 0;
    uint64_t usedCoreNum = 0;
    uint64_t totalUBSize = 0;

    // scatter
    uint64_t ncRound = 0;
    uint64_t ncRoundTail = 0;
    uint64_t totalRound = 0;
    uint64_t preCoreNum = 0;
};
#pragma pack()

inline void InitTilingData(uint8_t* tiling, AdaptiveMaxPool3DGradTilingData* const_data)
{
    uint64_t* src = (uint64_t*)tiling;
    uint64_t* dst = (uint64_t*)const_data;
    for (auto i = 0; i < sizeof(AdaptiveMaxPool3DGradTilingData) / 8; i++)
        *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
    AdaptiveMaxPool3DGradTilingData tiling_data; \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float
#define DTYPE_GRAD float
#define DTYPE_ARGMAX int32_t
#define DTYPE_Y float
#endif // _ADA_MAX_POOL3d_GRAD_TILING_DEF_H_