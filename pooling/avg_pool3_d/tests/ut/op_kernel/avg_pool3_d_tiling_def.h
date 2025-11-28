/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __AVG_POOL3_D_TILING_DEF_H__
#define __AVG_POOL3_D_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct AvgPool3DTilingData {
    uint64_t inN = 0;
    uint64_t inC = 0;
    uint64_t tileC = 0;
    uint64_t inD = 0;
    uint64_t inH = 0;
    uint64_t inW = 0;
    uint64_t outD = 0;
    uint64_t outH = 0;
    uint64_t outW = 0;
    uint64_t kD = 0;
    uint64_t kH = 0;
    uint64_t kW = 0;
    uint64_t dD = 0;
    uint64_t dH = 0;
    uint64_t dW = 0;
    uint64_t pD = 0;
    uint64_t pH = 0;
    uint64_t pW = 0;
    uint64_t divisorOverride = 0;
    uint64_t countIncludePad = 0;
    uint64_t ceilMode = 0;
    uint64_t formerLength = 0;
    uint64_t formerNum = 0;
    uint64_t tailLength = 0;
    uint64_t tailNum = 0;
    uint64_t indexBufLen = 0;
    uint64_t windowWNum = 0;
    uint64_t tileInput = 0;
    uint64_t tileHW = 0;
    uint64_t atomicAddNum = 0;
    uint64_t useCoreNum = 0;
    uint64_t ncFactor = 0;
    uint64_t doFactor = 0;
    uint64_t hoFactor = 0;
    uint64_t woFactor = 0;
    uint64_t ncOuter = 0;
    uint64_t doOuter = 0;
    uint64_t hoOuter = 0;
    uint64_t woOuter = 0;
    uint64_t ncTail = 0;
    uint64_t doTail = 0;
    uint64_t hoTail = 0;
    uint64_t woTail = 0;

    uint64_t blockFactor = 0;
    uint64_t blockTail = 0;
    uint64_t totalIdx = 0;
    uint64_t coreNums = 0;
    uint64_t usedCoreNum = 0;
};
#pragma pack()

inline void InitAvgPool3DTilingData(uint8_t* tiling, AvgPool3DTilingData* const_data)
{
    uint64_t *src = (uint64_t *)tiling;
    uint64_t *dst = (uint64_t *)const_data;
    for (auto i = 0; i < sizeof(AvgPool3DTilingData) / 8; i++) *(dst + i) = *(src + i);
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
AvgPool3DTilingData tiling_data; \
InitAvgPool3DTilingData(tiling_arg, &tiling_data)

#endif // __AVG_POOL3_D_TILING_DEF_H__
