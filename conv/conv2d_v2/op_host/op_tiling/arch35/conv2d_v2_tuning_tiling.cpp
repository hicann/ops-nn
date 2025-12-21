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
 * \file conv2d_v2_tuning_tiling.cc
 * \brief
 */

#include "conv2d_v2_tuning_tiling.h"
#include "log/log.h"
#include "nlohmann/json.hpp"
#include "register/tuning_bank_key_registry.h"

namespace tuningtiling {
void GetAttrsInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs)
{
  auto attrs = context->GetAttrs();
  const gert::ContinuousVector *stridesList = nullptr;
  const gert::ContinuousVector *padsList = nullptr;
  const gert::ContinuousVector *dilationsList = nullptr;
  const int64_t *groups = nullptr;
  size_t idx = 0;

  stridesList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  padsList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  dilationsList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
  groups = attrs->GetAttrPointer<int64_t>(idx++);

  conv2dArgs->groups = static_cast<int32_t>(*groups);
  const int64_t *stridesListData = reinterpret_cast<const int64_t *>(stridesList->GetData());
  const int64_t *padsListData = reinterpret_cast<const int64_t *>(padsList->GetData());
  const int64_t *dilationsListData = reinterpret_cast<const int64_t *>(dilationsList->GetData());

  conv2dArgs->aDtype = context->GetInputDesc(tuningtiling::INPUT_A_INDEX)->GetDataType();
  conv2dArgs->aFormat = context->GetInputDesc(tuningtiling::INPUT_A_INDEX)->GetOriginFormat();
  conv2dArgs->cDtype = context->GetInputDesc(tuningtiling::INPUT_C_INDEX)->GetDataType();
  conv2dArgs->cFormat = context->GetOutputDesc(tuningtiling::INPUT_C_INDEX)->GetOriginFormat();

  auto aShape = context->GetInputShape(0);
  auto &aOriShape = aShape->GetOriginShape();

  conv2dArgs->aShapeN = aOriShape.GetDim(tuningtiling::FORMAT_NCHW_N_INDEX);
  conv2dArgs->aShapeH = aOriShape.GetDim(tuningtiling::FORMAT_NCHW_H_INDEX);
  conv2dArgs->aShapeW = aOriShape.GetDim(tuningtiling::FORMAT_NCHW_W_INDEX);

  auto cShape = context->GetOutputShape(0);
  auto &cOriShape = cShape->GetOriginShape();

  conv2dArgs->cShapeH = cOriShape.GetDim(tuningtiling::FORMAT_NCHW_H_INDEX);
  conv2dArgs->cShapeW = cOriShape.GetDim(tuningtiling::FORMAT_NCHW_W_INDEX);

  // data format is NCHW; 2, 3 is index of strides_h, strides_w and dilationH, dilationW
  conv2dArgs->strideH = static_cast<int32_t>(stridesListData[tuningtiling::FORMAT_NCHW_H_INDEX]);
  conv2dArgs->strideW = static_cast<int32_t>(stridesListData[tuningtiling::FORMAT_NCHW_W_INDEX]);
  conv2dArgs->dilationH = static_cast<int32_t>(dilationsListData[tuningtiling::FORMAT_NCHW_H_INDEX]);
  conv2dArgs->dilationW = static_cast<int32_t>(dilationsListData[tuningtiling::FORMAT_NCHW_W_INDEX]);

  // 0, 1, 2, 3 is pad index for padTop, pad_b, padLeft, padRight
  conv2dArgs->padTop = static_cast<int32_t>(padsListData[tuningtiling::PAD_TOP_INDEX]);
  conv2dArgs->padBottom = static_cast<int32_t>(padsListData[tuningtiling::PAD_BOTTOM_INDEX]);
  conv2dArgs->padLeft = static_cast<int32_t>(padsListData[tuningtiling::PAD_LEFT_INDEX]);
  conv2dArgs->padRight = static_cast<int32_t>(padsListData[tuningtiling::PAD_RIGHT_INDEX]);

  const int32_t *opImplMode = nullptr;
  if (conv2dArgs->aDtype == ge::DT_FLOAT && tuningtiling::opImplModeIdx < attrs->GetAttrNum()) {
      opImplMode = attrs->GetAttrPointer<int32_t>(tuningtiling::opImplModeIdx);
      if (opImplMode != nullptr) {
        // op_impl_mode_enum: 1: default 2: high_performance 4: high_precision 8: super_performance
        // 16: support_of_bound_index  32: enable_float_32_execution  64: enable_hi_float_32_execution
        conv2dArgs->hf32Flag = (*opImplMode & 0x40) ? 1 : 0;
      }
  }
}

void GetBiasInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs,
                 size_t biasInputIndex)
{
  conv2dArgs->biasFlag = context->GetOptionalInputShape(biasInputIndex) != nullptr;
  if (conv2dArgs->biasFlag) {
    conv2dArgs->biasDtype = context->GetOptionalInputDesc(biasInputIndex)->GetDataType();
  } else {
    conv2dArgs->biasDtype = ge::DT_FLOAT16;
  }
}

void GetFilterInfo(const gert::TilingContext *context, std::shared_ptr<Conv2DV2InputArgs> &conv2dArgs,
                   size_t filterInputIndex)
{
  auto filterDesc = context->GetInputDesc(filterInputIndex);
  conv2dArgs->bDtype = filterDesc->GetDataType();

  auto filterOriFormat = filterDesc->GetOriginFormat();
  conv2dArgs->bFormat = filterOriFormat;

  auto filterShape = context->GetInputShape(filterInputIndex);
  auto &filterOriShape = filterShape->GetOriginShape();

  // filter format is NCHW; 0, 1, 2, 3 is dim index for filter_n, filter_c, filter_h, filter_w
  conv2dArgs->bShapeN = filterOriShape.GetDim(tuningtiling::FORMAT_NCHW_N_INDEX);
  conv2dArgs->bShapeC = filterOriShape.GetDim(tuningtiling::FORMAT_NCHW_C_INDEX);
  conv2dArgs->bShapeH = filterOriShape.GetDim(tuningtiling::FORMAT_NCHW_H_INDEX);
  conv2dArgs->bShapeW = filterOriShape.GetDim(tuningtiling::FORMAT_NCHW_W_INDEX);
}

std::string DisplayInfoDict(std::shared_ptr<void> &inputArgs, size_t size, std::string opType)
{
  auto &func = OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
  auto iter = func.find(ge::AscendString(opType.c_str()));
  const auto &parseFunc = iter->second.GetBankKeyParseFuncV2();
  ge::AscendString infoDictJsonAs;
  parseFunc(inputArgs, size, infoDictJsonAs);
  std::string info_dict_json_str(infoDictJsonAs.GetString(), infoDictJsonAs.GetLength());
  return info_dict_json_str;
}

bool TilingForConv2DV2Input(const gert::TilingContext *context, std::shared_ptr<void> &inputArgs, size_t &size)
{
  OP_CHECK_IF(context == nullptr, OP_LOGE("CANNKB", "context is nullptr."), return false);
  std::shared_ptr<Conv2DV2InputArgs> conv2dArgs = std::make_shared<Conv2DV2InputArgs>();
  OP_CHECK_IF(conv2dArgs == nullptr, OP_LOGE("CANNKB", "conv2dArgs is nullptr."), return false);
  GetFilterInfo(context, conv2dArgs, tuningtiling::INPUT_B_INDEX);
  GetBiasInfo(context, conv2dArgs, tuningtiling::INPUT_BIAS_INDEX);
  GetAttrsInfo(context, conv2dArgs);
  inputArgs = conv2dArgs;
  size = sizeof(Conv2DV2InputArgs);

  std::string opType = "Conv2DV2";
  OP_LOGD("CANNKB", "Op Conv2DInput query repo by info dict: %s", DisplayInfoDict(inputArgs, size, opType).c_str());

  return true;
}

DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv2DV2, Conv2DV2InputArgs, aDtype, bDtype, cDtype, biasDtype, aShapeN,
                                 aShapeH, aShapeW, bShapeN, bShapeC, bShapeH, bShapeW, cShapeH, cShapeW,
                                 aFormat, bFormat, cFormat, groups, strideH, strideW, dilationH,
                                 dilationW, padTop, padBottom, padLeft, padRight, biasFlag, hf32Flag,
                                 reserverdParam1, reserverdParam2, reserverdParam3, reserverdParam4,
                                 reserverdParam5, reserverdParam6);

REGISTER_OP_BANK_KEY_CONVERT_FUN_V2(Conv2DV2, TilingForConv2DV2Input);
REGISTER_TUNING_TILING_CLASS(Conv2DV2, Conv2DV2TunnerTiling);
}  // namespace tuningtiling