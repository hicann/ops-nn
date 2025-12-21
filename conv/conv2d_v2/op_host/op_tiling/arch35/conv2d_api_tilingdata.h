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
 * \file conv2d_api_tilingdata.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV2D_API_TILINGDATA_H
#define ASCENDC_TIKCFW_TILING_CONV2D_API_TILINGDATA_H

#include "register/tilingdata_base.h"

// sizeof(TConv2DTiling) must algin with 8Byte, type sort
namespace optiling {
BEGIN_TILING_DATA_DEF(TConv2DTiling)
TILING_DATA_FIELD_DEF(uint64_t, orgHi);
TILING_DATA_FIELD_DEF(uint64_t, orgWi);
TILING_DATA_FIELD_DEF(uint64_t, orgHo);
TILING_DATA_FIELD_DEF(uint64_t, orgWo);
TILING_DATA_FIELD_DEF(uint64_t, orgHixWi);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);
TILING_DATA_FIELD_DEF(uint32_t, orgCi);
TILING_DATA_FIELD_DEF(uint32_t, orgCo);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCi);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCo);
TILING_DATA_FIELD_DEF(uint32_t, hoL1);
TILING_DATA_FIELD_DEF(uint32_t, woL1);
TILING_DATA_FIELD_DEF(uint32_t, kAL1);
TILING_DATA_FIELD_DEF(uint32_t, kBL1);
TILING_DATA_FIELD_DEF(uint32_t, khL1);
TILING_DATA_FIELD_DEF(uint32_t, kwL1);
TILING_DATA_FIELD_DEF(uint32_t, nBL1);
TILING_DATA_FIELD_DEF(uint32_t, hoL0);
TILING_DATA_FIELD_DEF(uint32_t, woL0);
TILING_DATA_FIELD_DEF(uint32_t, kL0);
TILING_DATA_FIELD_DEF(uint32_t, nL0);
TILING_DATA_FIELD_DEF(uint32_t, pBufferFlag);
TILING_DATA_FIELD_DEF(uint32_t, groups);
TILING_DATA_FIELD_DEF(uint32_t, enlarge);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroups);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroupOpt);
TILING_DATA_FIELD_DEF(uint32_t, bUbNStep);
TILING_DATA_FIELD_DEF(uint32_t, bUbKStep);
TILING_DATA_FIELD_DEF(uint32_t, khUb);
TILING_DATA_FIELD_DEF(uint32_t, kwUb);
TILING_DATA_FIELD_DEF(uint32_t, kernelHxkernelW);
TILING_DATA_FIELD_DEF(uint32_t, kernelHxkernelWxkernelD);
TILING_DATA_FIELD_DEF(uint32_t, aL1SpaceSize);
TILING_DATA_FIELD_DEF(uint32_t, multiNBL1);
TILING_DATA_FIELD_DEF(uint32_t, cinAInCore);
TILING_DATA_FIELD_DEF(uint32_t, cinATailInCore);
TILING_DATA_FIELD_DEF(uint32_t, cinBInCore);
TILING_DATA_FIELD_DEF(uint32_t, cinBTailInCore);
TILING_DATA_FIELD_DEF(uint32_t, mStep);
TILING_DATA_FIELD_DEF(uint32_t, kStep);
TILING_DATA_FIELD_DEF(uint32_t, nStep);
TILING_DATA_FIELD_DEF(uint32_t, fmapKStride);
TILING_DATA_FIELD_DEF(uint32_t, weightKStride);
TILING_DATA_FIELD_DEF(uint32_t, cinOffsetBlockInGM);
TILING_DATA_FIELD_DEF(uint32_t, coutOffsetBlock);
TILING_DATA_FIELD_DEF(uint32_t, nL1DivBlockSize);
TILING_DATA_FIELD_DEF(uint32_t, kernelH);
TILING_DATA_FIELD_DEF(uint32_t, kernelW);
TILING_DATA_FIELD_DEF(uint32_t, strideH);
TILING_DATA_FIELD_DEF(uint32_t, strideW);
TILING_DATA_FIELD_DEF(uint32_t, dilationH);
TILING_DATA_FIELD_DEF(uint32_t, dilationW);
TILING_DATA_FIELD_DEF(uint32_t, padTop);
TILING_DATA_FIELD_DEF(uint32_t, padBottom);
TILING_DATA_FIELD_DEF(uint32_t, padLeft);
TILING_DATA_FIELD_DEF(uint32_t, padRight);
TILING_DATA_FIELD_DEF(uint32_t, innerBatch);
TILING_DATA_FIELD_DEF(uint8_t, iterateMNOrder);
TILING_DATA_FIELD_DEF(uint8_t, biasFullLoadFlag);
TILING_DATA_FIELD_DEF(uint8_t, fixpParamsFullLoadFlag);
TILING_DATA_FIELD_DEF(uint8_t, hf32Enable);
TILING_DATA_FIELD_DEF(uint8_t, hf32TransMode);
TILING_DATA_FIELD_DEF(uint8_t, hasBias);
TILING_DATA_FIELD_DEF(uint8_t, hasScale);
TILING_DATA_FIELD_DEF(uint8_t, dualOutput);
TILING_DATA_FIELD_DEF(uint8_t, quantMode0);
TILING_DATA_FIELD_DEF(uint8_t, reluMode0);
TILING_DATA_FIELD_DEF(uint8_t, clipMode0);
TILING_DATA_FIELD_DEF(uint8_t, quantMode1);
TILING_DATA_FIELD_DEF(uint8_t, reluMode1);
TILING_DATA_FIELD_DEF(uint8_t, clipMode1);
TILING_DATA_FIELD_DEF(int8_t, offsetx);
TILING_DATA_FIELD_DEF(int8_t, roundMode);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TConv2DTilingOp, TConv2DTiling);
}
#endif