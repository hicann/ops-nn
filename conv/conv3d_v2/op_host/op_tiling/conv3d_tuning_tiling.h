/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_TUNING_TILING_H_
#define CONV3D_TUNING_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
#pragma pack(push, 1)
struct Conv3DInputArgs {
  ge::DataType aDtype;
  ge::DataType bDtype;
  ge::DataType cDtype;
  ge::DataType biasDtype;
  uint64_t aShapeN;
  uint64_t aShapeD;
  uint64_t aShapeH;
  uint64_t aShapeW;
  uint64_t bShapeN;
  uint64_t bShapeC;
  uint64_t bShapeD;
  uint64_t bShapeH;
  uint64_t bShapeW;
  uint64_t cShapeD;
  uint64_t cShapeH;
  uint64_t cShapeW;
  ge::Format aFormat;
  ge::Format bFormat;
  ge::Format cFormat;
  uint64_t groups;
  uint64_t strideD;
  uint64_t strideH;
  uint64_t strideW;
  uint64_t dilationD;
  uint64_t dilationH;
  uint64_t dilationW;
  uint64_t padHead;
  uint64_t padTail;
  uint64_t padTop;
  uint64_t padBottom;
  uint64_t padLeft;
  uint64_t padRight;
  bool biasFlag;
};
#pragma pack(pop)
BEGIN_TUNING_TILING_DEF(Conv3DTunerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint64_t, groups);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgDo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgCo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgHo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgWo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgCi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgDi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgHi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgWi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelD);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelH);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelW);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, strideD);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, strideH);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, strideW);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, dilationD);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, dilationH);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, dilationW);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padHead);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padTail);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padTop);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padBottom);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padLeft);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, padRight);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, mL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kAL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, mAL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, pBufferFlag);
TUNING_TILING_DATA_FIELD_DEF(int8_t, offsetx);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, bl1FullLoad);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, al1FullLoad);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, bl1BypassFlag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, iterateMNOrder);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, biasFullLoadFlag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, fixpParamsFullLoadFlag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hf32Enable);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hf32TransMode);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, batchDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, nDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, mDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, doDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, groupDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, reserved1);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, reserved2);
END_TUNING_TILING_DEF
DECLARE_SCHEMA(Conv3DTunerTiling,
  FIELD(Conv3DTunerTiling, groups),
  FIELD(Conv3DTunerTiling, singleCoreDo),
  FIELD(Conv3DTunerTiling, singleCoreCo),
  FIELD(Conv3DTunerTiling, singleCoreM),
  FIELD(Conv3DTunerTiling, orgDo),
  FIELD(Conv3DTunerTiling, orgCo),
  FIELD(Conv3DTunerTiling, orgHo),
  FIELD(Conv3DTunerTiling, orgWo),
  FIELD(Conv3DTunerTiling, orgCi),
  FIELD(Conv3DTunerTiling, orgDi),
  FIELD(Conv3DTunerTiling, orgHi),
  FIELD(Conv3DTunerTiling, orgWi),
  FIELD(Conv3DTunerTiling, kernelD),
  FIELD(Conv3DTunerTiling, kernelH),
  FIELD(Conv3DTunerTiling, kernelW),
  FIELD(Conv3DTunerTiling, strideD),
  FIELD(Conv3DTunerTiling, strideH),
  FIELD(Conv3DTunerTiling, strideW),
  FIELD(Conv3DTunerTiling, dilationD),
  FIELD(Conv3DTunerTiling, dilationH),
  FIELD(Conv3DTunerTiling, dilationW),
  FIELD(Conv3DTunerTiling, padHead),
  FIELD(Conv3DTunerTiling, padTail),
  FIELD(Conv3DTunerTiling, padTop),
  FIELD(Conv3DTunerTiling, padBottom),
  FIELD(Conv3DTunerTiling, padLeft),
  FIELD(Conv3DTunerTiling, padRight),
  FIELD(Conv3DTunerTiling, mL0),
  FIELD(Conv3DTunerTiling, kL0),
  FIELD(Conv3DTunerTiling, nL0),
  FIELD(Conv3DTunerTiling, kAL1),
  FIELD(Conv3DTunerTiling, kBL1),
  FIELD(Conv3DTunerTiling, nBL1),
  FIELD(Conv3DTunerTiling, mAL1),
  FIELD(Conv3DTunerTiling, pBufferFlag),
  FIELD(Conv3DTunerTiling, offsetx),
  FIELD(Conv3DTunerTiling, bl1FullLoad),
  FIELD(Conv3DTunerTiling, al1FullLoad),
  FIELD(Conv3DTunerTiling, bl1BypassFlag),
  FIELD(Conv3DTunerTiling, iterateMNOrder),
  FIELD(Conv3DTunerTiling, biasFullLoadFlag),
  FIELD(Conv3DTunerTiling, fixpParamsFullLoadFlag),
  FIELD(Conv3DTunerTiling, hf32Enable),
  FIELD(Conv3DTunerTiling, hf32TransMode),
  FIELD(Conv3DTunerTiling, batchDim),
  FIELD(Conv3DTunerTiling, nDim),
  FIELD(Conv3DTunerTiling, mDim),
  FIELD(Conv3DTunerTiling, doDim),
  FIELD(Conv3DTunerTiling, groupDim),
  FIELD(Conv3DTunerTiling, reserved1),
  FIELD(Conv3DTunerTiling, reserved2));
}  // namespace tuningtiling
#endif