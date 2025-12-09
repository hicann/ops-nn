/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef __MASKED_SOFTMAX_WITH_REL_POS_BIAS_TILING_H__
#define __MASKED_SOFTMAX_WITH_REL_POS_BIAS_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct MaskedSoftmaxWithRelPosBiasTilingData {
    uint32_t tailStartCoreIdx;
    uint32_t singleCoreSize;
    uint32_t w;
    uint32_t n;
    uint32_t s1;
    uint32_t s2;
    uint32_t wns1s2;
    uint32_t wns1;
    uint32_t ws1s2;
    uint32_t ns1s2;
    uint32_t ws1;
    uint32_t ns1;
    uint32_t s1s2;
    uint32_t wns1s2Aligned;
    uint32_t s1s2Aligned;
    uint32_t ns1s2Aligned;
    uint32_t s2Aligned;
    uint32_t s2DtypeSize;
    uint32_t inQueueLen;
    uint32_t maskQueueLen;
    uint32_t attenBiasNum;
    uint32_t stackNum;
    uint32_t loopNum;
    uint32_t loopTailNum;
    uint32_t xCopyEleNum;
    uint32_t tailXCopyEleNum;
    float scaleValue;
    uint32_t castTempSize;
    uint32_t tmpXubSize;
    uint32_t tmpMaskUbSize;
    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling tailSoftmaxTilingData;
};
#pragma pack()

inline void InitMaskedSoftmaxWithRelPosBiasTilingData(
    uint8_t* tiling, MaskedSoftmaxWithRelPosBiasTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(MaskedSoftmaxWithRelPosBiasTilingData));
}

#define GET_TILING_DATA(tiling_data, tiling_arg)       \
    MaskedSoftmaxWithRelPosBiasTilingData tiling_data; \
    InitMaskedSoftmaxWithRelPosBiasTilingData(tiling_arg, &tiling_data)
#endif
