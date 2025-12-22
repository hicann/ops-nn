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
 * \file conv2d_v2_tiling_def.h
 * \brief
 */

#ifndef _GE_CONV2DV2_TILING_H_
#define _GE_CONV2DV2_TILING_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#define __forceinline__
#define FORMAT_X FORMAT_NCHW
#define FORMAT_FILTER FORMAT_NCHW
#define FORMAT_Y FORMAT_NCHW

#pragma pack(1)

struct Conv2DRunInfo {
    uint64_t hin = 0;
    uint64_t win = 0;
    uint64_t hout = 0;
    uint64_t wout = 0;
    uint32_t batch = 0;
    uint32_t cin = 0;
    uint32_t cout = 0;
    uint32_t kh = 0;
    uint32_t kw = 0;
    uint32_t batchDim = 0;
    uint32_t groupDim = 0;
    uint32_t nDim = 0;
    uint32_t hoDim = 0;
    uint32_t woDim = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t padTop = 0;
    uint32_t padLeft = 0;
    uint32_t groups = 0;
    uint32_t enlarge = 0;
    uint32_t cinOpt = 0;
    uint32_t coutOpt = 0;
    uint32_t groupOpt = 0;
    uint8_t hasBias = 0;
};

struct TConv2DTiling {
    uint64_t orgHi = 0;
    uint64_t orgWi = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    uint32_t orgCi = 0;
    uint32_t orgCo = 0;
    uint32_t singleCoreCi = 0;
    uint32_t singleCoreCo = 0;
    uint32_t hoL1 = 0;
    uint32_t woL1 = 0;
    uint32_t kAL1 = 0;
    uint32_t kBL1 = 0;
    uint32_t nBL1 = 0;
    uint32_t hoL0 = 0;
    uint32_t woL0 = 0;
    uint32_t kL0 = 0;
    uint32_t nL0 = 0;
    uint32_t pBufferFlag = 0;
    uint32_t groups = 0;
    uint32_t enlarge = 0;
    uint32_t singleCoreGroups = 0;
    uint32_t singleCoreGroupOpt = 0;
    uint32_t bUbNStep = 0;
    uint32_t bUbKStep = 0;
    uint32_t orgHixWi = 0;
    uint32_t kernelHxkernelW = 0;
    uint32_t kernelHxkernelWxkernelD = 0;
    uint32_t aL1SpaceSize = 0;
    uint32_t multiNBL1 = 0;
    uint32_t cinAInCore = 0;
    uint32_t cinATailInCore = 0;
    uint32_t cinBInCore = 0;
    uint32_t cinBTailInCore = 0;
    uint32_t mStep = 0;
    uint32_t kStep = 0;
    uint32_t nStep = 0;
    uint32_t fmapKStride = 0;
    uint32_t weightKStride = 0;
    uint32_t cinOffsetBlockInGM = 0;
    uint32_t coutOffsetBlock = 0;
    uint32_t nL1DivBlockSize = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t innerBatch = 1;
    uint8_t iterateMNOrder = 0;
    uint8_t biasFullLoadFlag = 0;
    uint8_t fixpParamsFullLoadFlag = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
    uint8_t hasBias = 0;
    uint8_t hasScale = 0;
    uint8_t dualOutput = 0;
    uint8_t quantMode0 = 0;
    uint8_t reluMode0 = 0;
    uint8_t clipMode0 = 0;
    uint8_t quantMode1 = 0;
    uint8_t reluMode1 = 0;
    uint8_t clipMode1 = 0;
    int8_t offsetx = 0;
    int8_t roundMode = 0;
};

struct Conv2DTilingData
{
    TConv2DTiling conv2dApiTiling;
    Conv2DRunInfo conv2dRunInfo;
};

#pragma pack()

inline void InitTilingData(uint8_t* tiling, Conv2DTilingData* constData) {
    memcpy(constData, tiling, sizeof(Conv2DTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    Conv2DTilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)
#endif