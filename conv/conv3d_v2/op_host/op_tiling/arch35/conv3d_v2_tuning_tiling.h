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
 * \file conv3d_v2_tuning_tiling.h
 * \brief
 */

#ifndef CONV3D_V2_TUNING_V2_TILING_H_
#define CONV3D_V2_TUNING_V2_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
constexpr uint32_t POS_INDEX_0 = 0;
constexpr uint32_t POS_INDEX_1 = 1;
constexpr uint32_t POS_INDEX_2 = 2;
constexpr uint32_t POS_INDEX_3 = 3;
constexpr uint32_t POS_INDEX_4 = 4;
constexpr uint32_t POS_INDEX_5 = 5;

#pragma pack(push, 1)
struct Conv3DV2InputArgs {
  ge::DataType aDtype;
  ge::DataType bDtype;
  ge::DataType cDtype;
  ge::DataType biasDtype;
  uint64_t aShapeN;
  uint32_t aShapeD;
  uint64_t aShapeH;
  uint64_t aShapeW;
  uint64_t bShapeN;
  uint64_t bShapeC;
  uint64_t bShapeD;
  uint64_t bShapeH;
  uint64_t bShapeW;
  uint32_t cShapeD;
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
  uint64_t reserverdParam1;
  uint64_t reserverdParam2;
  uint64_t reserverdParam3;
  float reserverdParam4;
  float reserverdParam5;
  float reserverdParam6;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv3DV2TunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint64_t, groups);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreDo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCi);
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
TUNING_TILING_DATA_FIELD_DEF(uint64_t, hoL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, woL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kAL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, hoL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, woL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, pBufferFlag);
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
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hoDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, doDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, groupDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, isC04Flag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, mMode);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv3DV2TunnerTiling,
  FIELD(Conv3DV2TunnerTiling, groups),
  FIELD(Conv3DV2TunnerTiling, singleCoreDo),
  FIELD(Conv3DV2TunnerTiling, singleCoreCo),
  FIELD(Conv3DV2TunnerTiling, singleCoreHo),
  FIELD(Conv3DV2TunnerTiling, singleCoreWo),
  FIELD(Conv3DV2TunnerTiling, singleCoreCi),
  FIELD(Conv3DV2TunnerTiling, orgDo),
  FIELD(Conv3DV2TunnerTiling, orgCo),
  FIELD(Conv3DV2TunnerTiling, orgHo),
  FIELD(Conv3DV2TunnerTiling, orgWo),
  FIELD(Conv3DV2TunnerTiling, orgCi),
  FIELD(Conv3DV2TunnerTiling, orgDi),
  FIELD(Conv3DV2TunnerTiling, orgHi),
  FIELD(Conv3DV2TunnerTiling, orgWi),
  FIELD(Conv3DV2TunnerTiling, kernelD),
  FIELD(Conv3DV2TunnerTiling, kernelH),
  FIELD(Conv3DV2TunnerTiling, kernelW),
  FIELD(Conv3DV2TunnerTiling, strideD),
  FIELD(Conv3DV2TunnerTiling, strideH),
  FIELD(Conv3DV2TunnerTiling, strideW),
  FIELD(Conv3DV2TunnerTiling, dilationD),
  FIELD(Conv3DV2TunnerTiling, dilationH),
  FIELD(Conv3DV2TunnerTiling, dilationW),
  FIELD(Conv3DV2TunnerTiling, padHead),
  FIELD(Conv3DV2TunnerTiling, padTail),
  FIELD(Conv3DV2TunnerTiling, padTop),
  FIELD(Conv3DV2TunnerTiling, padBottom),
  FIELD(Conv3DV2TunnerTiling, padLeft),
  FIELD(Conv3DV2TunnerTiling, padRight),
  FIELD(Conv3DV2TunnerTiling, hoL0),
  FIELD(Conv3DV2TunnerTiling, woL0),
  FIELD(Conv3DV2TunnerTiling, kL0),
  FIELD(Conv3DV2TunnerTiling, nL0),
  FIELD(Conv3DV2TunnerTiling, kAL1),
  FIELD(Conv3DV2TunnerTiling, kBL1),
  FIELD(Conv3DV2TunnerTiling, nBL1),
  FIELD(Conv3DV2TunnerTiling, hoL1),
  FIELD(Conv3DV2TunnerTiling, woL1),
  FIELD(Conv3DV2TunnerTiling, pBufferFlag),
  FIELD(Conv3DV2TunnerTiling, bl1FullLoad),
  FIELD(Conv3DV2TunnerTiling, al1FullLoad),
  FIELD(Conv3DV2TunnerTiling, bl1BypassFlag),
  FIELD(Conv3DV2TunnerTiling, iterateMNOrder),
  FIELD(Conv3DV2TunnerTiling, biasFullLoadFlag),
  FIELD(Conv3DV2TunnerTiling, fixpParamsFullLoadFlag),
  FIELD(Conv3DV2TunnerTiling, hf32Enable),
  FIELD(Conv3DV2TunnerTiling, hf32TransMode),
  FIELD(Conv3DV2TunnerTiling, batchDim),
  FIELD(Conv3DV2TunnerTiling, nDim),
  FIELD(Conv3DV2TunnerTiling, hoDim),
  FIELD(Conv3DV2TunnerTiling, doDim),
  FIELD(Conv3DV2TunnerTiling, groupDim),
  FIELD(Conv3DV2TunnerTiling, isC04Flag),
  FIELD(Conv3DV2TunnerTiling, mMode));

void GetAttrsInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3d_args);
void GetOutputInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3d_args);
void GetFmapInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3d_args,
                 size_t fmap_input_index);
void GetBiasInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3d_args,
                 size_t bias_input_index);
void GetFilterInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3d_args,
                   size_t filter_input_index);
bool TilingForConv3DV2Input(const gert::TilingContext *context, std::shared_ptr<void> &input_args,
                          size_t &size);
}  // namespace tuningtiling

#endif