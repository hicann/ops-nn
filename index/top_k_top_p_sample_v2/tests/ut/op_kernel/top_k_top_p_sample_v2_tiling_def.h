/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _TOP_K_TOP_P_SAMPLE_V2_TILING_H_
#define _TOP_K_TOP_P_SAMPLE_V2_TILING_H_

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

struct TopKTopPSampleV2TilingData {
    uint32_t numCore;
    uint32_t rowNum;
    uint32_t rowLen;
    uint32_t headCoreNum;
    uint32_t perHeadCoreRowNum;
    uint32_t tailCoreRowNum;
    uint32_t perHeadCorePartNum;
    uint32_t tailCorePartNum;
    uint32_t innerLoopEle;
    uint32_t innerLoopTime;
    uint32_t innerLoopEleTail;
    uint32_t innerLoopEleTailPad;
    uint32_t softmaxLoopTime;
    uint32_t softmaxLoopEleTail;
    uint32_t softmaxLoopEleTailPad;
    uint32_t eightKPartNum;
    uint32_t eightKPartTail;
    uint32_t eightKPartTailPad;
    uint32_t mrgMode;
    uint32_t headOffset;
    uint32_t isNeedLogits;
    float eps;
    uint32_t topKGuess;
    uint32_t ksMAX;
    uint32_t inputIsLogits;
    uint32_t isNeedSampleResult;
};

inline void InitTopKTopPSampleV2TilingData(uint8_t* tiling, TopKTopPSampleV2TilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(TopKTopPSampleV2TilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg) \
TopKTopPSampleV2TilingData tiling_data; \
InitTopKTopPSampleV2TilingData(tiling_arg, &tiling_data);
#endif