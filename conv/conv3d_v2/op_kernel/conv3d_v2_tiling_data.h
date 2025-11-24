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
 * \file conv3d_v2_tiling_data.h
 * \brief
 */
#ifndef CONV3D_V2_TILING_DATA_H
#define CONV3D_V2_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace Ops {
namespace NN {
namespace Conv3dV2 {

#pragma pack(push, 8)

#define CONV3D_COMMON_ATTRS \
    uint32_t strideH = 0; \
    uint32_t strideW = 0; \
    uint32_t strideD = 0; \
    uint32_t dilationH = 0; \
    uint32_t dilationW = 0; \
    uint32_t dilationD = 0; \
    uint32_t padHead = 0; \
    uint32_t padTail = 0; \
    uint32_t padTop = 0; \
    uint32_t padBottom = 0; \
    uint32_t padLeft = 0; \
    uint32_t padRight = 0

struct TConv3DTiling {
    uint64_t orgDo = 0;
    uint64_t orgHo = 0;
    uint64_t orgWo = 0;
    uint64_t orgDi = 0;
    uint64_t orgHi = 0;
    uint64_t orgWi = 0;
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreDo = 0;
    uint64_t singleCoreM = 0;
    uint64_t singleCoreWo = 0;
    uint64_t singleCoreHo = 0;
    uint64_t kL0xorgCoAlignN0 = 0;
    uint64_t kernelHxkernelW = 0;
    uint64_t cin1xOriHixOriWixk0 = 0;
    uint64_t oriHixOriWixk0 = 0;
    uint64_t oriWixk0 = 0;
    uint64_t orgHixWi = 0;
    uint64_t orgHoxWo = 0;
    uint32_t orgCi = 0;
    uint32_t kernelD = 0;
    uint32_t kernelH = 0;
    uint32_t kernelW = 0;
    uint32_t singleCoreCo = 0;
    uint32_t orgCo = 0;
    uint32_t singleCoreCi = 0;
    uint32_t singleCoreGroups = 0;
    uint32_t singleCoreGroupOpt = 0;
    uint32_t groups = 0;
    uint32_t enlarge = 0;
    CONV3D_COMMON_ATTRS;
    uint32_t mL0 = 0;
    uint32_t woL0 = 0;
    uint32_t kL0 = 0;
    uint32_t nL0 = 0;
    uint32_t kAL1 = 0;
    uint32_t kAL1Tail = 0;
    uint32_t kBL1 = 0;
    uint32_t kBL1Tail = 0;
    uint32_t nBL1 = 0;
    uint32_t mAL1 = 0;
    uint32_t woL1 = 0;
    uint32_t hoL0 = 0;
    uint32_t hoL1 = 0;
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
    uint32_t cinAInCore = 0;
    uint32_t cinATailInCore = 0;
    uint32_t nL0xk0 = 0;
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
    uint32_t kernelHxkernelWxkernelD = 0;
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
    uint8_t outputOrder = 0; // 0=M_Mode, 1=HW_Mode
};

struct Conv3DRunInfo {
    uint32_t batch = 0;
    uint32_t cin = 0;
    uint32_t din = 0;
    uint64_t hin = 0;
    uint64_t win = 0;
    uint32_t cout = 0;
    uint32_t kd = 0;
    uint32_t kh = 0;
    uint32_t kw = 0;
    uint32_t dout = 0;
    uint64_t hout = 0;
    uint64_t wout = 0;
    uint32_t batchDim = 0;
    uint32_t doDim = 0;
    uint32_t mDim = 0;
    uint32_t wDim = 0;
    uint32_t nDim = 0;
    uint32_t groupDim = 0;
    uint32_t hoDim = 0;
    CONV3D_COMMON_ATTRS;
    uint32_t groups = 0;
    uint32_t enlarge = 0;
    uint32_t cinOpt = 0;
    uint32_t coutOpt = 0;
    uint32_t groupOpt = 0;
    uint8_t hasBias = 0;
};

struct Conv3DV2TilingData {
    TConv3DTiling conv3dApiTiling;
    Conv3DRunInfo conv3dRunInfo;
};

#pragma pack(pop)

} // namespace Conv3dV2
} // namespace NN
} // namespace Ops

#endif // CONV3D_V2_TILING_DATA_H
