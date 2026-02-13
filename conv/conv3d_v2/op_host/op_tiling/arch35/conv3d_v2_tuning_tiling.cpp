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
 * \file conv3d_v2_tuning_v2_tiling.cc
 * \brief
 */

#include "conv3d_v2_tuning_tiling.h"
#include "log/log.h"
#include "nlohmann/json.hpp"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
void GetAttrsInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3dArgs)
{
  auto attrs = context->GetAttrs();
  size_t idx = 0;

  const gert::ContinuousVector *stridesList = nullptr;
  stridesList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const gert::ContinuousVector *padsList = nullptr;
  padsList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const gert::ContinuousVector *dilationsList = nullptr;
  dilationsList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  const int64_t *groups = nullptr;
  groups = attrs->GetAttrPointer<int64_t>(idx++);

  conv3dArgs->groups = static_cast<int32_t>(*groups);
  const int64_t *stridesListData = reinterpret_cast<const int64_t *>(stridesList->GetData());
  const int64_t *padsListData = reinterpret_cast<const int64_t *>(padsList->GetData());
  const int64_t *dilationsListData = reinterpret_cast<const int64_t *>(dilationsList->GetData());

  if (context->GetOutputDesc(0)->GetOriginFormat() == ge::FORMAT_NCDHW) {
    // data format is NCDHW; 2, 3 is index of strides_d, strides_h, strides_w and dilationD, dilationH, dilationW
    conv3dArgs->strideD = static_cast<int32_t>(stridesListData[POS_INDEX_2]);
    conv3dArgs->strideH = static_cast<int32_t>(stridesListData[POS_INDEX_3]);
    conv3dArgs->strideW = static_cast<int32_t>(stridesListData[POS_INDEX_4]);
    conv3dArgs->dilationD = static_cast<int32_t>(dilationsListData[POS_INDEX_2]);
    conv3dArgs->dilationH = static_cast<int32_t>(dilationsListData[POS_INDEX_3]);
    conv3dArgs->dilationW = static_cast<int32_t>(dilationsListData[POS_INDEX_4]);
  } else {
    // data format is NDHWC
    conv3dArgs->strideD = static_cast<int32_t>(stridesListData[POS_INDEX_1]);
    conv3dArgs->strideH = static_cast<int32_t>(stridesListData[POS_INDEX_2]);
    conv3dArgs->strideW = static_cast<int32_t>(stridesListData[POS_INDEX_3]);
    conv3dArgs->dilationD = static_cast<int32_t>(dilationsListData[POS_INDEX_1]);
    conv3dArgs->dilationH = static_cast<int32_t>(dilationsListData[POS_INDEX_2]);
    conv3dArgs->dilationW = static_cast<int32_t>(dilationsListData[POS_INDEX_3]);
  }

  // 0, 1, 2, 3, 4, 5 is pad index for pad_h, pad_t, pad_u, pad_b, pad_l, pad_r
  conv3dArgs->padHead = static_cast<int32_t>(padsListData[POS_INDEX_0]);
  conv3dArgs->padTail = static_cast<int32_t>(padsListData[POS_INDEX_1]);
  conv3dArgs->padTop = static_cast<int32_t>(padsListData[POS_INDEX_2]);
  conv3dArgs->padBottom = static_cast<int32_t>(padsListData[POS_INDEX_3]);
  conv3dArgs->padLeft = static_cast<int32_t>(padsListData[POS_INDEX_4]);
  conv3dArgs->padRight = static_cast<int32_t>(padsListData[POS_INDEX_5]);
}

void GetOutputInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3dArgs)
{
  auto outputDesc = context->GetOutputDesc(POS_INDEX_0);
  conv3dArgs->cDtype = outputDesc->GetDataType();

  auto outputOriFormat = outputDesc->GetOriginFormat();
  conv3dArgs->cFormat = outputOriFormat;

  auto outputShape = context->GetOutputShape(POS_INDEX_0);
  auto &outputOriShape = outputShape->GetOriginShape();

  if (outputOriFormat == ge::FORMAT_NCDHW) {
    // output format is NCDHW; 2, 3, 4 is dim index for dedx_d, dedx_h, dedx_w
    conv3dArgs->cShapeD = static_cast<uint32_t>(outputOriShape.GetDim(POS_INDEX_2));
    conv3dArgs->cShapeH = outputOriShape.GetDim(POS_INDEX_3);
    conv3dArgs->cShapeW = outputOriShape.GetDim(POS_INDEX_4);
  } else {
    // output format is NDHWC
    conv3dArgs->cShapeD = static_cast<uint32_t>(outputOriShape.GetDim(POS_INDEX_1));
    conv3dArgs->cShapeH = outputOriShape.GetDim(POS_INDEX_2);
    conv3dArgs->cShapeW = outputOriShape.GetDim(POS_INDEX_3);
  }
}

void GetFmapInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3dArgs,
                 size_t fmapInputIndex)
{
  auto fmapDesc = context->GetInputDesc(fmapInputIndex);
  conv3dArgs->aDtype = fmapDesc->GetDataType();

  auto fmapOriFormat = fmapDesc->GetOriginFormat();
  conv3dArgs->aFormat = fmapOriFormat;

  auto fmapShape = context->GetInputShape(fmapInputIndex);
  auto &fmapOriShape = fmapShape->GetOriginShape();

  if (fmapOriFormat == ge::FORMAT_NCDHW) {
    // dedy format is NCDHW; 0, 2, 3, 4 is dim index for dedy_n, dedy_d, dedy_h, dedy_w
    conv3dArgs->aShapeN = fmapOriShape.GetDim(POS_INDEX_0);
    conv3dArgs->aShapeD = static_cast<uint32_t>(fmapOriShape.GetDim(POS_INDEX_2));
    conv3dArgs->aShapeH = fmapOriShape.GetDim(POS_INDEX_3);
    conv3dArgs->aShapeW = fmapOriShape.GetDim(POS_INDEX_4);
  } else {
    // NDHWC
    conv3dArgs->aShapeN = fmapOriShape.GetDim(POS_INDEX_0);
    conv3dArgs->aShapeD = static_cast<uint32_t>(fmapOriShape.GetDim(POS_INDEX_1));
    conv3dArgs->aShapeH = fmapOriShape.GetDim(POS_INDEX_2);
    conv3dArgs->aShapeW = fmapOriShape.GetDim(POS_INDEX_3);
  }
}

void GetBiasInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3dArgs,
                 size_t biasInputIndex)
{
  conv3dArgs->biasFlag = context->GetOptionalInputShape(biasInputIndex) != nullptr;
  if (conv3dArgs->biasFlag) {
    conv3dArgs->biasDtype = context->GetOptionalInputDesc(biasInputIndex)->GetDataType();
  } else {
    conv3dArgs->biasDtype = ge::DT_FLOAT16;
  }
}

void GetFilterInfo(const gert::TilingContext *context, std::shared_ptr<Conv3DV2InputArgs> &conv3dArgs,
                   size_t filterInputIndex)
{
  auto filterDesc = context->GetInputDesc(filterInputIndex);
  conv3dArgs->bDtype = filterDesc->GetDataType();

  auto filterOriFormat = filterDesc->GetOriginFormat();
  conv3dArgs->bFormat = filterOriFormat;

  auto filterShape = context->GetInputShape(filterInputIndex);
  auto &filterOriShape = filterShape->GetOriginShape();

  if (filterOriFormat == ge::FORMAT_NCDHW) {
    // filter format is NCDHW; 0, 1, 2, 3, 4 is dim index for filter_n, filter_c, filter_d, filter_h, filter_w
    conv3dArgs->bShapeN = filterOriShape.GetDim(POS_INDEX_0);
    conv3dArgs->bShapeC = filterOriShape.GetDim(POS_INDEX_1);
    conv3dArgs->bShapeD = filterOriShape.GetDim(POS_INDEX_2);
    conv3dArgs->bShapeH = filterOriShape.GetDim(POS_INDEX_3);
    conv3dArgs->bShapeW = filterOriShape.GetDim(POS_INDEX_4);
  } else if (filterOriFormat == ge::FORMAT_DHWCN) {
    // filter format is DHWCN
    conv3dArgs->bShapeN = filterOriShape.GetDim(POS_INDEX_4);
    conv3dArgs->bShapeC = filterOriShape.GetDim(POS_INDEX_3);
    conv3dArgs->bShapeD = filterOriShape.GetDim(POS_INDEX_0);
    conv3dArgs->bShapeH = filterOriShape.GetDim(POS_INDEX_1);
    conv3dArgs->bShapeW = filterOriShape.GetDim(POS_INDEX_2);
  } else {
    // filter format is NDHWC
    conv3dArgs->bShapeN = filterOriShape.GetDim(POS_INDEX_0);
    conv3dArgs->bShapeC = filterOriShape.GetDim(POS_INDEX_4);
    conv3dArgs->bShapeD = filterOriShape.GetDim(POS_INDEX_1);
    conv3dArgs->bShapeH = filterOriShape.GetDim(POS_INDEX_2);
    conv3dArgs->bShapeW = filterOriShape.GetDim(POS_INDEX_3);
  }
}

bool TilingForConv3DV2Input(const gert::TilingContext *context, std::shared_ptr<void> &inputArgs,
                                     size_t &size)
{
  OP_CHECK_IF(context == nullptr, OP_LOGE("CANNKB", "context is nullptr."), return false);
  std::shared_ptr<Conv3DV2InputArgs> conv3dArgs = std::make_shared<Conv3DV2InputArgs>();
  OP_CHECK_IF(conv3dArgs == nullptr, OP_LOGE("CANNKB","conv3dArgs is nullptr."), return false);
  GetFilterInfo(context, conv3dArgs, POS_INDEX_1);
  GetFmapInfo(context, conv3dArgs, POS_INDEX_0);
  GetBiasInfo(context, conv3dArgs, POS_INDEX_2);
  GetOutputInfo(context, conv3dArgs);
  GetAttrsInfo(context, conv3dArgs);
  inputArgs = conv3dArgs;
  size = sizeof(Conv3DV2InputArgs);

  return true;
}

DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv3DV2, Conv3DV2InputArgs, aDtype, bDtype, cDtype, biasDtype, aShapeN, aShapeD,
                              aShapeH, aShapeW, bShapeN, bShapeC, bShapeD, bShapeH, bShapeW, cShapeD,
                              cShapeH, cShapeW, aFormat, bFormat, cFormat, groups, strideD,
                              strideH, strideW, dilationD, dilationH, dilationW, padHead, padTail, padTop,
                              padBottom, padLeft, padRight, biasFlag, reserverdParam1, reserverdParam2,
                              reserverdParam3, reserverdParam4, reserverdParam5, reserverdParam6);

REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(Conv3DV2, TilingForConv3DV2Input);
REGISTER_TUNING_TILING_CLASS(Conv3DV2, Conv3DV2TunnerTiling);
}  // namespace tuningtiling