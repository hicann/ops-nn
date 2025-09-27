/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "avgpool2d_conv2d_backprop_input_util.h"
#include "../../../../conv/convolution_backward/op_host/op_api/convolutionbackward.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "op_api/op_api_def.h"
#include "level0/dilation.h"
#include "level0/fill.h"
#include "level0/reduce_sum_op.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "runtime/context.h"

using namespace op;
namespace avgpool2d_conv2d_input_util {

static const int kHDimNC1HWC0Idx = 2;
static const int kWDimNC1HWC0Idx = 3;
static const int kHDimNCHWIdx = 2;
static const int kWDimNCHWIdx = 3;
static const int nDimNCLIdx = 0;
static const int cDimNCLIdx = 1;
static const int lDimNCLIdx = 2;
static const int coDimCoCiDHWIdx = 0;
static const int ciDimCoCiDHWIdx = 1;
static const int nDimNCDHWIdx = 0;
static const int cDimNCDHWIdx = 1;
// 工具中-------------------------
static const int dDimNCDHWIdx = 2;
static const int hDimNCDHWIdx = 3;
static const int wDimNCDHWIdx = 4;
static const int32_t kHDimNDHWCIdx = 2;
static const int32_t kWDimNDHWCIdx = 3;
// -----------------------------
static const int kDILATIONHIdx = 0;
static const int kDILATIONWIdx = 1;
static const int kSTRIDEHIdx = 0;
static const int kSTRIDEWIdx = 1;
static const int kPADDINGUPIdx = 0;
static const int kPADDINGLEFTIdx = 1;
static const int kPadding4UpIdx = 0;
static const int kPadding4DownIdx = 1;
static const int kPadding4LeftIdx = 2;
static const int kPadding4RightIdx = 3;

static const int NUM2 = 2;
static const int NUM4 = 4;


static bool IsPreInsertDilation(const ConvolutionBackwardInputTensorForAvgPool2d &inputTensor,
                                const ConvolutionBackwardParamsForAvgPool2d &params) {
  // In order to prevent the workspaceSize from being too large, dilation will not be inserted with special case.
  op::Shape gradOutputSpecialShape = op::Shape({8, 1280, 64, 64});
  op::Shape inputSpecialShape = op::Shape({8, 3, 896, 896});
  op::Shape weightSpecialShape = op::Shape({1280, 3, 14, 14});
  bool isPreInserDiationSpecialFlag = (inputTensor.gradOutput->GetViewShape() == gradOutputSpecialShape &&
                                       inputTensor.input->GetViewShape() == inputSpecialShape &&
                                       inputTensor.weight->GetViewShape() == weightSpecialShape &&
                                       ((*params.stride)[kSTRIDEHIdx] == 14 && (*params.stride)[kSTRIDEWIdx] == 14));
  // When the stride == 2, using set2d yields superior performance, so there is no need for predilation
  return (inputTensor.weight->GetViewShape().GetDim(kHDimNCHWIdx) > 1 ||
          inputTensor.weight->GetViewShape().GetDim(kWDimNCHWIdx) > 1) &&
         ((*params.stride)[kSTRIDEHIdx] > NUM2 || (*params.stride)[kSTRIDEWIdx] > NUM2) && !isPreInserDiationSpecialFlag;
}

static bool IsInputSupportInsertDilation() {
  // 判断当前SOC版本是否支持前后插Dilation
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
    OP_LOGD("The soc version is Ascend910B or ASCEND910_93, ConvolutionBackward supports dilation");
    return true;
  }
  return false;
}

static bool IsPostInsertDilation(const aclTensor *weight, const ConvolutionBackwardParamsForAvgPool2d &params) {
  return (weight->GetViewShape().GetDim(kHDimNCHWIdx) == 1 && weight->GetViewShape().GetDim(kWDimNCHWIdx) == 1) &&
         ((*params.stride)[kSTRIDEHIdx] > 1 || (*params.stride)[kSTRIDEWIdx] > 1);
}

static const aclTensor *PreDilation(ConvolutionBackwardInputTensorForAvgPool2d &inputTensor, ConvolutionBackwardParamsForAvgPool2d &params,
                                    aclOpExecutor *executor) {
  const aclTensor *preDilationGradOutputNC1HWC0 = nullptr;
  int64_t preDilationDilationsVector[] = {1, 1, (*params.stride)[kSTRIDEHIdx], (*params.stride)[kSTRIDEWIdx], 1};
  /* 当输出的h/w为1时，把传给Dilation算子的dilation值修正为fmap_h/w，避免在dilation值超大时，Dilation算子超时 */
  if (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEHIdx] > inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx)) {
    preDilationDilationsVector[kHDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx);
  }
  if (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEWIdx] > inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx)) {
    preDilationDilationsVector[kWDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx);
  }
  aclIntArray *preDilationDilations = executor->AllocIntArray(preDilationDilationsVector, 5);
  CHECK_RET(preDilationDilations != nullptr, nullptr);
  int64_t preDilationPadUp = 0;
  int64_t preDilationPadLeft = 0;
  int64_t preDilationPadH = 0;
  int64_t preDilationPadW = 0;
  if (params.padding->Size() == NUM4) {
    preDilationPadH = (*params.padding)[kPadding4UpIdx] + (*params.padding)[kPadding4DownIdx];
    preDilationPadW = (*params.padding)[kPadding4LeftIdx] + (*params.padding)[kPadding4RightIdx];
  } else {
    preDilationPadH = NUM2 * (*params.padding)[kPADDINGUPIdx];
    preDilationPadW = NUM2 * (*params.padding)[kPADDINGLEFTIdx];
  }
  int64_t preDilationPadDown =
      inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx) -
      (inputTensor.weight->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.dilation)[kDILATIONHIdx] +
      preDilationPadH -
      (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEHIdx] - 1;
  int64_t preDilationPadRight =
      inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx) -
      (inputTensor.weight->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.dilation)[kDILATIONWIdx] +
      preDilationPadW -
      (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEWIdx] - 1;

  int64_t preDilationPadsVector[] = {preDilationPadUp, preDilationPadDown, preDilationPadLeft, preDilationPadRight};
  aclIntArray *preDilationPads = executor->AllocIntArray(preDilationPadsVector, 4);
  CHECK_RET(preDilationPads != nullptr, nullptr);

  float paddingValue = 0;

  preDilationGradOutputNC1HWC0 =
      l0op::Dilation(inputTensor.gradOutput, preDilationDilations, preDilationPads, paddingValue, executor);
  return preDilationGradOutputNC1HWC0;
}

inline bool IsCubeSupportHf32() {
    if (op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910B &&
        op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_93 &&
        op::GetCurrentPlatformInfo().GetSocVersion() != op::SocVersion::ASCEND910_95) {
        return false;
    }
    return true;
}

// 根据promoteType + cubeMathType 判断是否要走HF32分支
static bool Avg_NeedCubeGoHF32(const DataType cubeTensorPromoteType, int8_t cubeMathType) {
    // USE_HF32场景，如果promoteType为BF16或FP16时，提示不支持该选项
    constexpr int8_t USE_HF32 = 3;
    constexpr int8_t ALLOW_FP32_DOWN_PRECISION = 1;

    if (cubeMathType == USE_HF32) {
        if (cubeTensorPromoteType == DataType::DT_BF16) {
            OP_LOGW("The cubeMathType cann't be set to USE_HF32 when the dtype is BF16.");
        }
        if (cubeTensorPromoteType == DataType::DT_FLOAT16) {
            OP_LOGW("The cubeMathType cann't be set to USE_HF32 when the dtype is FP16.");
        }
    }

    // Ascend910B + promoteType为FP32 + ALLOW_DOWN / USE_HF32 才需要走HF32分支
    if (IsCubeSupportHf32() && (cubeTensorPromoteType == DataType::DT_FLOAT) &&
        (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_HF32)) {
        return true;
    }
    return false;
}

static bool ConvBackGoHf32(const ConvolutionBackwardInputTensorForAvgPool2d& inputTensor, int8_t cubeMathType) {
  auto promoteType = op::PromoteType(inputTensor.input->GetDataType(), inputTensor.weight->GetDataType());
  promoteType = op::PromoteType(promoteType, inputTensor.gradOutput->GetDataType());
  return Avg_NeedCubeGoHF32(promoteType, cubeMathType);
}

static const aclTensor *PerformConv2DBackpropInput(ConvolutionBackwardInputTensorForAvgPool2d &inputTensor,
                                            ConvolutionBackwardParamsForAvgPool2d &params, aclOpExecutor *executor) {
  const aclTensor *gradInputNC1HWC0 = nullptr;
  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
      gradInputNC1HWC0 =
        l0op::Conv2DBackpropInput(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                  params.padding, params.dilation, params.groups, executor,
                                  useHf32, inputTensor.weight->GetDataType());
      return gradInputNC1HWC0;
  }
  // other platform
  if (useHf32) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                      params.padding, params.dilation, params.groups, executor);
  } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputFp322Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  } else if (inputTensor.weight->GetDataType() == DataType::DT_BF16) {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputBf162Bf16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  } else {
    gradInputNC1HWC0 =
        l0op::Conv2DBackpropInputFp162Fp16(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                           params.padding, params.dilation, params.groups, executor);
  }
  return gradInputNC1HWC0;
}

static const aclTensor *PostDilation(const aclTensor *dxGradInputNC1HWC0, ConvolutionBackwardInputTensorForAvgPool2d &inputTensor,
                                     ConvolutionBackwardParamsForAvgPool2d &params, aclOpExecutor *executor) {
  const aclTensor *postDilationGradInputNC1HWC0 = nullptr;
  int64_t postDilationDilationsVector[] = {1, 1, (*params.stride)[kSTRIDEHIdx], (*params.stride)[kSTRIDEWIdx], 1};
  /* 当输出的h/w为1时，把传给Dilation算子的dilation值修正为fmap_h/w，避免在dilation值超大时，Dilation算子超时 */
  if (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEHIdx] > inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx)) {
    postDilationDilationsVector[kHDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx);
  }
  if (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) == 1 &&
      (*params.stride)[kSTRIDEWIdx] > inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx)) {
    postDilationDilationsVector[kWDimNC1HWC0Idx] = inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx);
  }
  aclIntArray *post_dilation_dilations = executor->AllocIntArray(postDilationDilationsVector, 5);
  CHECK_RET(post_dilation_dilations != nullptr, nullptr);

  int64_t postDilationPadUp = 0;
  int64_t postDilationPadLeft = 0;
  int64_t padUp = 0;
  int64_t padLeft = 0;
  if (params.padding->Size() == NUM4) {
    postDilationPadUp = -(*params.padding)[kPadding4UpIdx];
    postDilationPadLeft = -(*params.padding)[kPadding4LeftIdx];
    padUp = (*params.padding)[kPadding4UpIdx];
    padLeft = (*params.padding)[kPadding4LeftIdx];
  } else {
    postDilationPadUp = -(*params.padding)[kPADDINGUPIdx];
    postDilationPadLeft = -(*params.padding)[kPADDINGLEFTIdx];
    padUp = (*params.padding)[kPADDINGUPIdx];
    padLeft = (*params.padding)[kPADDINGLEFTIdx];
  }
  int64_t postDilationPadDown =
      inputTensor.input->GetViewShape().GetDim(kHDimNCHWIdx) + padUp -
      (inputTensor.gradOutput->GetViewShape().GetDim(kHDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEHIdx] - 1;
  int64_t postDilationPadRight =
      inputTensor.input->GetViewShape().GetDim(kWDimNCHWIdx) + padLeft -
      (inputTensor.gradOutput->GetViewShape().GetDim(kWDimNCHWIdx) - 1) * (*params.stride)[kSTRIDEWIdx] - 1;
  int64_t postDilationPadsVector[] = {postDilationPadUp, postDilationPadDown, postDilationPadLeft,
                                      postDilationPadRight};
  aclIntArray *postDilationPads = executor->AllocIntArray(postDilationPadsVector, 4);
  CHECK_RET(postDilationPads != nullptr, nullptr);
  float paddingValue = 0;

  postDilationGradInputNC1HWC0 =
      l0op::Dilation(dxGradInputNC1HWC0, post_dilation_dilations, postDilationPads, paddingValue, executor);
  return postDilationGradInputNC1HWC0;
}

const aclTensor *CalculateConv2DBackpropInputForAvgPool2d(ConvolutionBackwardInputTensorForAvgPool2d &inputTensor,
                                                     ConvolutionBackwardParamsForAvgPool2d &params, aclOpExecutor *executor) {
  const aclTensor *dxGradInputNC1HWC0Res = nullptr;
  if (IsPreInsertDilation(inputTensor, params) && IsInputSupportInsertDilation() &&
      inputTensor.input->GetDataType() != DataType::DT_BF16) {
    const aclTensor *preDilationGradOutputNC1HWC0 = nullptr;
    preDilationGradOutputNC1HWC0 = PreDilation(inputTensor, params, executor);
    int64_t newstrideVector[] = {1, 1};
    aclIntArray *newstride = executor->AllocIntArray(newstrideVector, 2);
    CHECK_RET(newstride != nullptr, nullptr);
    ConvolutionBackwardInputTensorForAvgPool2d newInputTensor = {preDilationGradOutputNC1HWC0, inputTensor.input,
                                                     inputTensor.weight};
    ConvolutionBackwardParamsForAvgPool2d newparams = {params.biasSizes, newstride,         params.padding,
                                           params.dilation,  params.transposed, params.outputPadding,
                                           params.groups,    params.outputMask, params.cubeMathType};
    dxGradInputNC1HWC0Res = PerformConv2DBackpropInput(newInputTensor, newparams, executor);
  } else if (IsPostInsertDilation(inputTensor.weight, params) && IsInputSupportInsertDilation() &&
             inputTensor.input->GetDataType() != DataType::DT_BF16) {
    const aclTensor *dxGradInputNC1HWC0 = nullptr;
    int64_t newstrideVector[] = {1, 1};
    int64_t newpaddingVector[] = {0, 0, 0, 0};
    aclIntArray *newstride = executor->AllocIntArray(newstrideVector, 2);
    CHECK_RET(newstride != nullptr, nullptr);
    // 4: newpaddingVector 数组分配的空间大小
    aclIntArray *newpadding = executor->AllocIntArray(newpaddingVector, 4);
    CHECK_RET(newpadding != nullptr, nullptr);
    op::Shape newInputStorageShape;
    op::Shape newInputOrishape;
    for (size_t i = 0; i < inputTensor.input->GetStorageShape().GetDimNum(); i++) {
      newInputStorageShape.AppendDim(i == kHDimNC1HWC0Idx || i == kWDimNC1HWC0Idx
                                         ? inputTensor.gradOutput->GetStorageShape().GetDim(i)
                                         : inputTensor.input->GetStorageShape().GetDim(i));}
    for (size_t i = 0; i < inputTensor.input->GetViewShape().GetDimNum(); i++) {
      newInputOrishape.AppendDim(i == kHDimNCHWIdx || i == kWDimNCHWIdx
                                     ? inputTensor.gradOutput->GetViewShape().GetDim(i)
                                     : inputTensor.input->GetViewShape().GetDim(i));}
    auto newInput =
        executor->AllocTensor(newInputStorageShape, newInputOrishape, inputTensor.weight->GetDataType(),
                              inputTensor.input->GetStorageFormat(), inputTensor.input->GetOriginalFormat());
    ConvolutionBackwardInputTensorForAvgPool2d newInputTensor = {inputTensor.gradOutput, newInput, inputTensor.weight};
    ConvolutionBackwardParamsForAvgPool2d newparams = {params.biasSizes, newstride,         newpadding,
                                           params.dilation,  params.transposed, params.outputPadding,
                                           params.groups,    params.outputMask, params.cubeMathType};
    dxGradInputNC1HWC0 = PerformConv2DBackpropInput(newInputTensor, newparams, executor);
    OP_CHECK(dxGradInputNC1HWC0 != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with PerformConv2DBackpropInput failed, Conv2dBackpropInput return nullptr."),
             return nullptr);
    dxGradInputNC1HWC0Res = PostDilation(dxGradInputNC1HWC0, inputTensor, params, executor);
  } else {
    dxGradInputNC1HWC0Res = PerformConv2DBackpropInput(inputTensor, params, executor);
  }
  OP_CHECK(dxGradInputNC1HWC0Res != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with dxGradInputNC1HWC0Res failed, return nullptr."),
           return nullptr);
  return dxGradInputNC1HWC0Res;
}
}
