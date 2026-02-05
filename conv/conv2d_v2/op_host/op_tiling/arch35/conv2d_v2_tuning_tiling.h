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
 * \file conv2d_v2_tuning_tiling.h
 * \brief
 */

#ifndef CONV2D_V2_TUNING_TILING_H_
#define CONV2D_V2_TUNING_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"
namespace tuningtiling {
constexpr uint32_t INPUT_A_INDEX = 0;
constexpr uint32_t INPUT_B_INDEX = 1;
constexpr uint32_t INPUT_C_INDEX = 0;
constexpr uint32_t INPUT_BIAS_INDEX = 2;
constexpr uint32_t opImplModeIdx = 6;

constexpr uint32_t FORMAT_NCHW_N_INDEX = 0;
constexpr uint32_t FORMAT_NCHW_C_INDEX = 1;
constexpr uint32_t FORMAT_NCHW_H_INDEX = 2;
constexpr uint32_t FORMAT_NCHW_W_INDEX = 3;

constexpr uint32_t PAD_TOP_INDEX = 0;
constexpr uint32_t PAD_BOTTOM_INDEX = 1;
constexpr uint32_t PAD_LEFT_INDEX = 2;
constexpr uint32_t PAD_RIGHT_INDEX = 3;

#pragma pack(push, 1)
struct Conv2DV2InputArgs {
  ge::DataType aDtype;
  ge::DataType bDtype;
  ge::DataType cDtype;
  ge::DataType biasDtype;
  int64_t aShapeN;
  int64_t aShapeH;
  int64_t aShapeW;
  int64_t bShapeN;
  int64_t bShapeC;
  int64_t bShapeH;
  int64_t bShapeW;
  int64_t cShapeH;
  int64_t cShapeW;
  ge::Format aFormat;
  ge::Format bFormat;
  ge::Format cFormat;
  int64_t groups;
  int64_t strideH;
  int64_t strideW;
  int64_t dilationH;
  int64_t dilationW;
  int64_t padTop;
  int64_t padBottom;
  int64_t padLeft;
  int64_t padRight;
  bool biasFlag;
  bool hf32Flag;
  uint64_t reserverdParam1;
  uint64_t reserverdParam2;
  uint64_t reserverdParam3;
  float reserverdParam4;
  float reserverdParam5;
  float reserverdParam6;
};
#pragma pack(pop)

#define TUNING_TILING_DATA_FIELD_DEF_WITH_INITVALUE(data_type, field_name, value)              \
  public:                                                                                      \
     data_type field_name = value;                                                             \
     FieldHandler field_name##_handler_ = FieldHandler(this, #data_type, #field_name)

BEGIN_TUNING_TILING_DEF(Conv2DV2TunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint64_t, groups);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgCi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgHi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgWi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgCo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgHo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, orgWo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelH);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelW);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCi);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreHo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreWo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, hoL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, woL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kAL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, khL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kwL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, multiNBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, hoL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, woL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nL0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, pBufferFlag);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kernelHxkernelW);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinAInCore);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinATailInCore);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, oriHinxWin);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinOffsetBlockInGM);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, mStep);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, fmapKStride);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nStep);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kStep);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, weightKStride);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, coutOffsetBlock);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinBInCore);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinBTailInCore);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nBL1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, nL1DivBlockSize);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, strideH);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, strideW);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dilationH);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, dilationW);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, padTop);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, padBottom);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, padLeft);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, padRight);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, iterateMNOrder);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, biasFullLoadFlag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, fixpParamsFullLoadFlag);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, offsetx);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hf32Enable);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hf32TransMode);
TUNING_TILING_DATA_FIELD_DEF_WITH_INITVALUE(uint8_t, isC04Flag, 0);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, roundMode);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, batchDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, hoDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, woDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, nDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, groupDim);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, mMode);
TUNING_TILING_DATA_FIELD_DEF_WITH_INITVALUE(uint32_t, innerBatch, 1);
TUNING_TILING_DATA_FIELD_DEF_WITH_INITVALUE(uint32_t, enlarge, 0);
TUNING_TILING_DATA_FIELD_DEF_WITH_INITVALUE(uint8_t, isWeightUbTransFlag, 0);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv2DV2TunnerTiling,
  FIELD(Conv2DV2TunnerTiling, groups),
  FIELD(Conv2DV2TunnerTiling, orgCi),
  FIELD(Conv2DV2TunnerTiling, orgHi),
  FIELD(Conv2DV2TunnerTiling, orgWi),
  FIELD(Conv2DV2TunnerTiling, orgCo),
  FIELD(Conv2DV2TunnerTiling, orgHo),
  FIELD(Conv2DV2TunnerTiling, orgWo),
  FIELD(Conv2DV2TunnerTiling, kernelH),
  FIELD(Conv2DV2TunnerTiling, kernelW),
  FIELD(Conv2DV2TunnerTiling, singleCoreCi),
  FIELD(Conv2DV2TunnerTiling, singleCoreCo),
  FIELD(Conv2DV2TunnerTiling, singleCoreHo),
  FIELD(Conv2DV2TunnerTiling, singleCoreWo),
  FIELD(Conv2DV2TunnerTiling, hoL1),
  FIELD(Conv2DV2TunnerTiling, woL1),
  FIELD(Conv2DV2TunnerTiling, kAL1),
  FIELD(Conv2DV2TunnerTiling, kBL1),
  FIELD(Conv2DV2TunnerTiling, khL1),
  FIELD(Conv2DV2TunnerTiling, kwL1),
  FIELD(Conv2DV2TunnerTiling, multiNBL1),
  FIELD(Conv2DV2TunnerTiling, hoL0),
  FIELD(Conv2DV2TunnerTiling, woL0),
  FIELD(Conv2DV2TunnerTiling, kL0),
  FIELD(Conv2DV2TunnerTiling, nL0),
  FIELD(Conv2DV2TunnerTiling, pBufferFlag),
  FIELD(Conv2DV2TunnerTiling, kernelHxkernelW),
  FIELD(Conv2DV2TunnerTiling, cinAInCore),
  FIELD(Conv2DV2TunnerTiling, cinATailInCore),
  FIELD(Conv2DV2TunnerTiling, oriHinxWin),
  FIELD(Conv2DV2TunnerTiling, cinOffsetBlockInGM),
  FIELD(Conv2DV2TunnerTiling, mStep),
  FIELD(Conv2DV2TunnerTiling, fmapKStride),
  FIELD(Conv2DV2TunnerTiling, nStep),
  FIELD(Conv2DV2TunnerTiling, kStep),
  FIELD(Conv2DV2TunnerTiling, weightKStride),
  FIELD(Conv2DV2TunnerTiling, coutOffsetBlock),
  FIELD(Conv2DV2TunnerTiling, cinBInCore),
  FIELD(Conv2DV2TunnerTiling, cinBTailInCore),
  FIELD(Conv2DV2TunnerTiling, nBL1),
  FIELD(Conv2DV2TunnerTiling, nL1DivBlockSize),
  FIELD(Conv2DV2TunnerTiling, strideH),
  FIELD(Conv2DV2TunnerTiling, strideW),
  FIELD(Conv2DV2TunnerTiling, dilationH),
  FIELD(Conv2DV2TunnerTiling, dilationW),
  FIELD(Conv2DV2TunnerTiling, padTop),
  FIELD(Conv2DV2TunnerTiling, padBottom),
  FIELD(Conv2DV2TunnerTiling, padLeft),
  FIELD(Conv2DV2TunnerTiling, padRight),
  FIELD(Conv2DV2TunnerTiling, iterateMNOrder),
  FIELD(Conv2DV2TunnerTiling, biasFullLoadFlag),
  FIELD(Conv2DV2TunnerTiling, fixpParamsFullLoadFlag),
  FIELD(Conv2DV2TunnerTiling, offsetx),
  FIELD(Conv2DV2TunnerTiling, hf32Enable),
  FIELD(Conv2DV2TunnerTiling, hf32TransMode),
  FIELD(Conv2DV2TunnerTiling, isC04Flag),
  FIELD(Conv2DV2TunnerTiling, roundMode),
  FIELD(Conv2DV2TunnerTiling, batchDim),
  FIELD(Conv2DV2TunnerTiling, hoDim),
  FIELD(Conv2DV2TunnerTiling, woDim),
  FIELD(Conv2DV2TunnerTiling, nDim),
  FIELD(Conv2DV2TunnerTiling, groupDim),
  FIELD(Conv2DV2TunnerTiling, mMode),
  FIELD(Conv2DV2TunnerTiling, innerBatch),
  FIELD(Conv2DV2TunnerTiling, enlarge),
  FIELD(Conv2DV2TunnerTiling, isWeightUbTransFlag));

void GetAttrsInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs);
void GetBiasInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs,
                 size_t biasInputIndex);
void GetFilterInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs,
                  size_t filterInputIndex);
std::string DisplayInfoDict(std::shared_ptr<void> &inputArgs, size_t size, std::string opType);
bool TilingForConv2DV2Input(const gert::TilingContext *context, std::shared_ptr<void> &inputArgs, size_t &size);
}  // namespace tuningtiling

#endif