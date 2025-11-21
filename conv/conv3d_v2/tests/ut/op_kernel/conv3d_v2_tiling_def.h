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
 * \file conv3d_v2_tiling_def.h
 * \brief
 */

#ifndef _CONV3D_V2_TILING_DEF_H_
#define _CONV3D_V2_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

struct Conv3DRunInfo {
    uint32_t batch;
    uint32_t din;
    uint32_t cin;
    uint64_t hin;
    uint64_t win;
    uint32_t cout;
    uint32_t kd;
    uint32_t kh;
    uint32_t kw;
    uint32_t dout;
    uint64_t hout;
    uint64_t wout;
    uint32_t batchDim;
    uint32_t doDim;
    uint32_t mDim;
    uint32_t wDim;
    uint32_t nDim;
    uint32_t groupDim;
    uint32_t strideH;
    uint32_t dilationH;
    uint32_t strideW;
    uint32_t dilationW;
    uint32_t strideD;
    uint32_t dilationD;
    uint32_t padHead;
    uint32_t padTail;
    uint32_t padTop;
    uint32_t padBottom;
    uint32_t padLeft;
    uint32_t padRight;
    uint32_t groups;
    uint32_t enlarge;
    uint32_t cinOpt;
    uint32_t coutOpt;
    uint32_t groupOpt;
    uint8_t hasBias;
};

struct TConv3DTiling {
    uint32_t groups = 0;
    uint64_t orgDo = 0;
    uint32_t orgCo = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint32_t orgCi = 0;
    uint64_t orgDi = 0;
    uint64_t orgHi = 0;
    uint64_t orgWi = 0;
    uint32_t kernelD = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint64_t singleCoreDo = 0;
    uint32_t singleCoreCo = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreWo = 0;
    uint32_t singleCoreGroups = 0;
    uint32_t singleCoreGroupOpt = 0;
    uint32_t enlarge = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t strideD = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t dilationD = 0;
    uint32_t padHead = 0;
    uint32_t padTail = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t mL0 = 0;
    uint32_t kL0 = 0;
    uint32_t nL0 = 0;
    uint32_t woL0 = 0;
    uint32_t kAL1 = 0;
    uint32_t kAL1Tail = 0;
    uint32_t kBL1 = 0;
    uint32_t kBL1Tail = 0;
    uint32_t nBL1 = 0;
    uint32_t mAL1 = 0;
    uint32_t woL1 = 0;
    uint32_t KBL1Divk0 = 0;
    uint32_t KBL1TailDivk0 = 0;
    uint32_t nBL1DivnL0 = 0;
    uint32_t mAL1DivmL0 = 0;
    uint32_t fmapKStride = 0;
    uint32_t weightKStride = 0;
    uint32_t cinOffsetBlockInGM = 0;
    uint32_t coutOffsetBlock = 0;
    uint32_t nL1DivBlockSize = 0;
    uint32_t cin1InAL1 = 0;
    uint32_t cin1InAL1Tail = 0;
    uint32_t cinBInCore = 0;
    uint32_t cinBTailInCore = 0;
    uint32_t nL0xk0 = 0;
    uint64_t kL0xorgCoAlignN0 = 0;
    uint64_t kernelHxkernelW = 0;
    uint64_t cin1xOriHixOriWixk0 = 0;
    uint64_t oriHixOriWixk0 = 0;
    uint64_t oriWixk0 = 0;
    uint64_t orgHixWi = 0;
    uint64_t orgHoxWo = 0;
    uint32_t mStep = 0;
    uint32_t kStep = 0;
    uint32_t nStep = 0;
    uint32_t aL1SpaceSize = 0;
    uint32_t multiNBL1 = 0;
    uint32_t pBufferFlag = 0;
    uint32_t groupOpt = 0;
    uint32_t cinOpt = 0;
    uint32_t coutOpt = 0;
    uint32_t mUB = 0;
    uint32_t nUB = 0;
    uint32_t scaleAndBiasLoadType = 0;
    uint32_t workspaceSize = 0;
    int8_t offsetx = 0;
    int8_t roundMode = 0;
    uint8_t hasBias = 0;
    uint8_t hasScale = 0;
    uint8_t bl1FullLoad = 0;
    uint8_t al1FullLoad = 0;
    uint8_t bl1BypassFlag = 0;
    uint8_t iterateMNOrder = 0;
    uint8_t biasFullLoadFlag = 0;
    uint8_t fixpParamsFullLoadFlag = 0;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
    uint8_t quantType = 0;

    uint8_t resvered1 = 0;
    uint8_t resvered2 = 0;
    uint8_t resvered3 = 0;
};

struct Conv3DTilingData
{
    TConv3DTiling conv3dApiTiling;
    Conv3DRunInfo conv3dRunInfo;
};


#pragma pack()

inline void InitTilingData(uint8_t* tiling, Conv3DTilingData* constData) {
    memcpy(constData, tiling, sizeof(Conv3DTilingData));
}

#define GET_TILING_DATA(tilingData, tilingArg) \
    Conv3DTilingData tilingData; \
    InitTilingData(tilingArg, &tilingData)
#endif // _CONV3D_V2_TILING_DEF_H_
