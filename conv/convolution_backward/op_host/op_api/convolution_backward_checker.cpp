/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "convolution_backward_checker.h"

using namespace op;
using namespace l0op;
using namespace Ops::NN;
#ifdef __cplusplus
extern "C" {
#endif
// 工具方法----------------------------------------------------------------------------------------------------------
bool CheckDtypeValid(const aclTensor *inputTensor, bool transposed) {
  const SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  // 检查输入aclTensor的数据类型是否在ConvolutionBackward支持列表内
  if (socVersion == SocVersion::ASCEND910_95) {
    auto dtypeSupportList = GetDtypeSupportListBySocVersion4ConvBackward(transposed);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  } else {
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  }
  return true;
}

bool CheckParamsValueAllZero(const aclIntArray *params) {
    if (params != nullptr) {
        for (uint64_t i = 0; i < params->Size(); ++i) {
            if ((*params)[i] != 0) {
                return false;
            }
        }
    }
    return true;
}

bool CheckFormatValid(const aclTensor *inputTensor, const string &tensorName) {
    // API在输入Tensor的Format为ND时, 仅支持输入Tensor的维度是3, 并当做NCL格式处理
    op::Format inputFormat = inputTensor->GetStorageFormat();
    auto inputDim = inputTensor->GetViewShape().GetDimNum();
    if (inputDim == CONV1DINPUTDIM) {
        OP_CHECK(inputFormat == op::Format::FORMAT_ND || inputFormat == op::Format::FORMAT_NCL,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 1D scenes, the %s format only supports ND and NCL.",
                        tensorName.c_str()), return false);
    } else if (inputDim == CONV2DINPUTDIM) {
        OP_CHECK(inputFormat == op::Format::FORMAT_NCHW,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 2D scenes, the %s format only supports NCHW.",
              tensorName.c_str()), return false);
    } else if (inputDim == CONV3DINPUTDIM) {
        OP_CHECK(inputFormat == op::Format::FORMAT_NCDHW,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "In 3D scenes, the %s format only supports NCDHW.",
              tensorName.c_str()), return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The %s tensor dimension of this API only supports 3~5 dimensions.",
                tensorName.c_str());
        return false;
    }
    return true;
}

bool CheckParamsValue(const aclIntArray *params, bool isPad) {
    int64_t minValue = (isPad) ? 0 : 1;
    if (params != nullptr) {
        for (uint64_t i = 0; i < params->Size(); ++i) {
            if ((*params)[i] < minValue) {
                return false;
            }
        }
    }
    return true;
}

void GetChannleIndex(const op::Shape &shape, const op::Format &format, int64_t &channelIndex) {
    auto inputDim = shape.GetDimNum();
    if (format == op::Format::FORMAT_NDHWC || format == op::Format::FORMAT_NHWC) {
      channelIndex = inputDim - 1;
    } else {
      channelIndex = 1;
    }
}

bool CheckResolutionGEKernelShape(const op::Shape &inputShape, const op::Shape &weightShape, const ConvolutionBackwardParams &params, int64_t dimIdx) {
  int64_t dimOrder = dimIdx - 2; // 2: the dim N and D
  int64_t filterDimDilation = (weightShape[dimIdx] - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputShape.GetDim(dimIdx) + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  bool dimInputExpect = dimInput >= 0;
  OP_CHECK(dimInputExpect,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "(in_dim(%ld) + pad_dim(%ld) * 2) should >= ((weight_shape(%ld) - 1) * dilation(%ld) + 1)",
      inputShape.GetDim(dimIdx), (*params.padding)[dimOrder], weightShape[dimIdx], (*params.dilation)[dimOrder]),
      return false);
  return true;
}

int64_t GetExpectNum(const op::Shape &inputShape, const op::Shape &weightShape, const ConvolutionBackwardParams &params, int64_t dimIdx) {
  int64_t dimOrder = dimIdx - 2; // 2: the dim N and D
  int64_t filterDimDilation = (weightShape[dimIdx] - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputShape.GetDim(dimIdx) + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  int64_t dimExpect = dimInput / (*params.stride)[dimOrder] + 1;
  return dimExpect;
}

bool GetExpectValueDHW_95(const ConvolutionBackwardInputTensor &inputTensor, const ConvolutionBackwardParams &params, struct ExpectValue &expectValue, const op::Shape &inputShape, const op::Shape &weightShape) {
      op::Format weightFormat = inputTensor.weight->GetStorageFormat();
      op::Format inputFormat = inputTensor.input->GetStorageFormat();
      int64_t inputDVal = 0;
      int64_t inputHVal = 0;
      int64_t inputWVal = 0;
      GetInputShapeSize(inputFormat, inputShape, inputDVal, inputHVal, inputWVal);
      int64_t weightDVal = 0;
      int64_t weightHVal = 0;
      int64_t weightWVal = 0;
      GetWeightShapeSize(weightFormat, weightShape, weightDVal, weightHVal, weightWVal);
      if (!CheckResolutionGEKernelShape_95(inputDVal, weightDVal, 0, params) ||
          !CheckResolutionGEKernelShape_95(inputHVal, weightHVal, 1, params) ||
          !CheckResolutionGEKernelShape_95(inputWVal, weightWVal, 2, params)) {  // 2 : the param W
          return false;
      }
      expectValue.doExpect = GetExpectNum_95(inputDVal, weightDVal, 0, params);
      expectValue.hoExpect = GetExpectNum_95(inputHVal, weightHVal, 1, params);
      expectValue.woExpect = GetExpectNum_95(inputWVal, weightWVal, 2, params);  // 2 : the param W
      return true;
}

void GetInputShapeSize(const op::Format &format, const op::Shape &shape, int64_t &shapeDVal, int64_t &shapeHVal, int64_t &shapeWVal) {
    if (format == op::Format::FORMAT_NCDHW) {
      shapeDVal = shape.GetDim(dDimNCDHWIdx);
      shapeHVal = shape.GetDim(hDimNCDHWIdx);
      shapeWVal = shape.GetDim(wDimNCDHWIdx);
    } else {
      shapeDVal = shape.GetDim(1);
      shapeHVal = shape.GetDim(kHDimNDHWCIdx);
      shapeWVal = shape.GetDim(kWDimNDHWCIdx);
    }
}

bool CheckResolutionGEKernelShape_95(int64_t inputVal, int64_t weightVal, int64_t dimOrder, const ConvolutionBackwardParams &params) {
  int64_t filterDimDilation = (weightVal - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputVal + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  bool dimInputExpect = dimInput >= 0;
  OP_CHECK(dimInputExpect,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "(in_dim(%ld) + pad_dim(%ld) * 2) should >= ((weight_shape(%ld) - 1) * dilation(%ld) + 1)",
      inputVal, (*params.padding)[dimOrder], weightVal, (*params.dilation)[dimOrder]),
      return false);
  return true;
}

int64_t GetExpectNum_95(int64_t inputVal, int64_t weightVal, int64_t dimOrder, const ConvolutionBackwardParams &params) {
  int64_t filterDimDilation = (weightVal - 1) * (*params.dilation)[dimOrder] + 1;
  int64_t dimInput = inputVal + (*params.padding)[dimOrder] * 2 - filterDimDilation; // 2 : pad two dim
  int64_t dimExpect = dimInput / (*params.stride)[dimOrder] + 1;
  return dimExpect;
}

string AclarrayToString(const aclIntArray *array) {
  string str = "";
  if (array == nullptr) {
    return str;
  }
  for (uint64_t i = 0; i < array->Size(); ++i) {
    str += to_string((*array)[i]);
    if (i < array->Size() - 1) {
      str += ",";
    }
  }
  return str;
}

void GetWeightShapeSize(const op::Format &weightFormat, const op::Shape &weightShape, int64_t &weightDVal, int64_t &weightHVal, int64_t &weightWVal) {
  if (weightFormat == op::Format::FORMAT_NCDHW) {
    weightDVal = weightShape[dDimNCDHWIdx];
    weightHVal = weightShape[hDimNCDHWIdx];
    weightWVal = weightShape[wDimNCDHWIdx];
  } else {
    weightDVal = weightShape[1];
    weightHVal = weightShape[kHDimNDHWCIdx];
    weightWVal = weightShape[kWDimNDHWCIdx];
  }
}

aclnnStatus CalculateConvolutionBackwardWithEmpty(ConvolutionBackwardInputTensor &inputTensor,
                                                  ConvolutionBackwardOutput &outputTensor,
                                                  ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // Index 为 1：进行gradWeight空tensor运算
  if ((*params.outputMask)[1] && outputTensor.gradWeight->Size() != 0) {
    auto weightContiguous = l0op::Contiguous(inputTensor.weight, executor);
    auto gradWeightZeros = l0op::ZerosLike(weightContiguous, executor);
    OP_CHECK(gradWeightZeros != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, weight with ZerosLike return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto result = l0op::ViewCopy(gradWeightZeros, outputTensor.gradWeight, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, weight with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }

  // Index 为 2：进行bias空tensor运算
  if ((*params.outputMask)[2] && outputTensor.gradBias->Size() != 0) {
    op::Shape biasGradShape = {(*params.biasSizes)[0]};
    auto biasTensor = executor->AllocTensor(biasGradShape, inputTensor.weight->GetDataType());
    biasTensor->SetStorageFormat(op::Format::FORMAT_ND);
    biasTensor->SetViewFormat(op::Format::FORMAT_ND);
    biasTensor->SetOriginalFormat(op::Format::FORMAT_ND);

    auto gradBiasZeros = l0op::ZerosLike(biasTensor, executor);
    OP_CHECK(gradBiasZeros != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, bias with ZerosLike return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(gradBiasZeros, outputTensor.gradBias, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, bias with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }

  return ACLNN_SUCCESS;
}
// -----------------------------------------------------------------------------------------------------------------
namespace Ops {
namespace NN {
namespace Conv {

bool ConvolutionBackwardChecker::CheckDataTypeValidForGradInput() {
  if (socVersion_ != SocVersion::ASCEND910_95) {
      return true;
  }
  OP_CHECK(outputTensor_.gradInput->GetDataType() == inputTensor_.input->GetDataType(),
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradInput data type[%s] should be equal to input data type[%s]",
      op::ToString(outputTensor_.gradInput->GetDataType()).GetString(),
      op::ToString(inputTensor_.input->GetDataType()).GetString()), return false);
  return true;
}

bool ConvolutionBackwardChecker::CheckDataTypeValidForGradWeight() {
  if (socVersion_ != SocVersion::ASCEND910_95) {
      return true;
  }
  if (inputTensor_.weight->GetDataType() == DataType::DT_HIFLOAT8 ||
    inputTensor_.weight->GetDataType() == DataType::DT_FLOAT8_E4M3FN) {
    return true;
  }

  OP_CHECK(outputTensor_.gradWeight->GetDataType() == inputTensor_.weight->GetDataType(),
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradWeight data type[%s] should be equal to weight data type[%s]",
      op::ToString(outputTensor_.gradWeight->GetDataType()).GetString(),
      op::ToString(inputTensor_.weight->GetDataType()).GetString()), return false);
  return true;
}

bool ConvolutionBackwardChecker::CheckDataTypeValidForGradBias() {
  if (socVersion_ != SocVersion::ASCEND910_95) {
      return true;
  }
  OP_CHECK(outputTensor_.gradBias->GetDataType() == inputTensor_.gradOutput->GetDataType(),
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradBias data type[%s] should be equal to gradOutput data type[%s]",
      op::ToString(outputTensor_.gradBias->GetDataType()).GetString(),
      op::ToString(inputTensor_.gradOutput->GetDataType()).GetString()), return false);
  return true;
}

bool ConvolutionBackwardChecker::CheckParamsValidForBpFilter8bit() {
  if (inputTensor_.input->GetDataType() != DataType::DT_HIFLOAT8 &&
    inputTensor_.input->GetDataType() != DataType::DT_FLOAT8_E4M3FN) {
    return true;
  }

  // check outputpadding for transposed
  if (params_.transposed) {
    OP_CHECK(CheckParamsValueAllZero(params_.outputPadding),
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When transpose is true and the input data type is %s or %s, "
      "the value of outputPadding[%s] must be all 0",
      op::ToString(DataType::DT_HIFLOAT8).GetString(),
      op::ToString(DataType::DT_FLOAT8_E4M3FN).GetString(),
      AclarrayToString(params_.outputPadding).c_str()), return false);
  }

  // check groups
  OP_CHECK(params_.groups == 1,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When outputMask[1] = True, %s dtype only supports groups = 1, currently is %ld",
      op::ToString(inputTensor_.input->GetDataType()).GetString(), params_.groups),
    return false);

  return true;
}

bool ConvolutionBackwardChecker::InterceptConvFor8bit()
{
  // transposed=true时且为dx算子支持8bit，transposed=false，dw,dx算子均不支持8bit
  if (params_.transposed && (*params_.outputMask)[0]) {
    return true;
  }

  if (IsConv8bit(inputTensor_.gradOutput->GetDataType()) || IsConv8bit(inputTensor_.input->GetDataType()) ||
      IsConv8bit(inputTensor_.weight->GetDataType()) || IsConv8bit(outputTensor_.gradInput->GetDataType()) ||
      IsConv8bit(outputTensor_.gradWeight->GetDataType())) {
    OP_LOGE(
        ACLNN_ERR_PARAM_INVALID,
        "The dtype of DT_HIFLOAT8 or DT_FLOAT8_E4M3FN is not supported now, "
        "currently gradOutput is %s, input is %s, weight is %s, gradInput is %s, "
        "gradWeight is %s. ",
        op::ToString(inputTensor_.gradOutput->GetDataType()).GetString(),
        op::ToString(inputTensor_.input->GetDataType()).GetString(),
        op::ToString(inputTensor_.weight->GetDataType()).GetString(),
        op::ToString(outputTensor_.gradInput->GetDataType()).GetString(),
        op::ToString(outputTensor_.gradWeight->GetDataType()).GetString());
    return false;
  }
  return true;
}

bool ConvolutionBackwardChecker::IsConv8bit(const DataType& dType) const {
  return dType == DataType::DT_HIFLOAT8 || dType == DataType::DT_FLOAT8_E4M3FN;
}

// -----------------------------------------------------------------------------------------------------------------
bool ConvolutionBackwardChecker::CheckDtypeValidFor8bit(const DataType& dType) {
  bool isGradOutput8bit = inputTensor_.gradOutput->GetDataType() == dType;
  bool isInput8bit = inputTensor_.input->GetDataType() == dType;
  bool isWeight8bit = inputTensor_.weight->GetDataType() == dType;
  bool isGradInput8bit = outputTensor_.gradInput->GetDataType() == dType;
  bool is8bitFlag = isGradOutput8bit || isInput8bit || isWeight8bit || isGradInput8bit;
  bool all8bitFlag = isGradOutput8bit && isInput8bit && isWeight8bit && isGradInput8bit;
  if (outputTensor_.gradBias != nullptr) {
    bool isGradBias8bit = outputTensor_.gradBias->GetDataType() == dType;
    is8bitFlag = is8bitFlag || isGradBias8bit;
    all8bitFlag = all8bitFlag && isGradBias8bit;
  }

  if (outputTensor_.gradBias != nullptr) {
    OP_CHECK(!is8bitFlag || all8bitFlag,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When any input or output data types is %s, all of them must be %s, "
      "currently gradOutput is %s, input is %s, weight is %s, gradInput is %s, gradBias is %s",
      op::ToString(dType).GetString(), op::ToString(dType).GetString(),
      op::ToString(inputTensor_.gradOutput->GetDataType()).GetString(),
      op::ToString(inputTensor_.input->GetDataType()).GetString(),
      op::ToString(inputTensor_.weight->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradInput->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradBias->GetDataType()).GetString()),
      return false);
  } else {
    OP_CHECK(!is8bitFlag || all8bitFlag,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When any input or output data types is %s, all of them must be %s, "
      "currently gradOutput is %s, input is %s, weight is %s, gradInput is %s",
      op::ToString(dType).GetString(), op::ToString(dType).GetString(),
      op::ToString(inputTensor_.gradOutput->GetDataType()).GetString(),
      op::ToString(inputTensor_.input->GetDataType()).GetString(),
      op::ToString(inputTensor_.weight->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradInput->GetDataType()).GetString()),
      return false);
  }
  return true;
}

bool ConvolutionBackwardChecker::CheckDtypeValidForBpFilter8bit(const DataType& dType) {
  bool isGradOutput8bit = inputTensor_.gradOutput->GetDataType() == dType;
  bool isInput8bit = inputTensor_.input->GetDataType() == dType;
  bool isWeight8bit = inputTensor_.weight->GetDataType() == dType;
  bool isGradWeight32bit = outputTensor_.gradWeight->GetDataType() == DataType::DT_FLOAT;
  bool is8bitFlag = isGradOutput8bit || isInput8bit || isWeight8bit;
  bool all8bitFlag = isGradOutput8bit && isInput8bit && isWeight8bit && isGradWeight32bit;
  if (outputTensor_.gradBias != nullptr) {
    bool isGradBias8bit = outputTensor_.gradBias->GetDataType() == dType;
    is8bitFlag = is8bitFlag || isGradBias8bit;
    all8bitFlag = all8bitFlag && isGradBias8bit;
  }

  if (outputTensor_.gradBias != nullptr) {
    OP_CHECK(!is8bitFlag || all8bitFlag,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When outputMask[1] = true, any input or gradBias data types is %s, "
      "all input and gradBias data types must be %s, and gradWeight data type must be %s, "
      "currently gradOutput is %s, input is %s, weight is %s, gradWeight is %s, gradBias is %s",
      op::ToString(dType).GetString(), op::ToString(dType).GetString(), op::ToString(DataType::DT_FLOAT).GetString(),
      op::ToString(inputTensor_.gradOutput->GetDataType()).GetString(),
      op::ToString(inputTensor_.input->GetDataType()).GetString(),
      op::ToString(inputTensor_.weight->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradWeight->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradBias->GetDataType()).GetString()),
      return false);
  } else {
    OP_CHECK(!is8bitFlag || all8bitFlag,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When outputMask[1] = true, any input data types is %s, "
      "all input data types must be %s, and gradWeight data type must be %s, "
      "currently gradOutput is %s, input is %s, weight is %s, gradWeight is %s",
      op::ToString(dType).GetString(), op::ToString(dType).GetString(), op::ToString(DataType::DT_FLOAT).GetString(),
      op::ToString(inputTensor_.gradOutput->GetDataType()).GetString(),
      op::ToString(inputTensor_.input->GetDataType()).GetString(),
      op::ToString(inputTensor_.weight->GetDataType()).GetString(),
      op::ToString(outputTensor_.gradWeight->GetDataType()).GetString()),
      return false);
  }

  bool isGradOutputFp8 = inputTensor_.gradOutput->GetDataType() == DataType::DT_FLOAT8_E4M3FN;
  bool isInputFp8 = inputTensor_.input->GetDataType() == DataType::DT_FLOAT8_E4M3FN;
  bool isWeightFp8 = inputTensor_.weight->GetDataType() == DataType::DT_FLOAT8_E4M3FN;
  bool isGradWeightFp8 = outputTensor_.gradWeight->GetDataType() == DataType::DT_FLOAT8_E4M3FN;
  bool isGradBiasFp8 = outputTensor_.gradBias->GetDataType() == DataType::DT_FLOAT8_E4M3FN;
  bool isFp8Flag = isGradOutputFp8 || isInputFp8 || isWeightFp8 || isGradWeightFp8 || isGradBiasFp8;
  OP_CHECK(!isFp8Flag,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When outputMask[1] = true, not support dataType of all input and output being DT_FLOAT8_E4M3FN now"), return false);
  return true;
}

bool ConvolutionBackwardChecker::CheckConvParams(size_t inputDim) {
    // stride >= 1
    OP_CHECK(CheckParamsValue(params_.stride, false),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of stride[%s] must be greater than or equal to 1.",
             AclarrayToString(params_.stride).c_str()),
             return false);

    // padding >= 0
    OP_CHECK(CheckParamsValue(params_.padding, true),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of padding[%s] must be greater than or equal to 0.",
             AclarrayToString(params_.padding).c_str()),
             return false);

    // dilation >= 1
    OP_CHECK(CheckParamsValue(params_.dilation, false),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of dilation[%s] must be greater than or equal to 1.",
             AclarrayToString(params_.dilation).c_str()),
             return false);
    // outputPadding >= 0
    if (params_.transposed) {
      OP_CHECK(CheckParamsValue(params_.outputPadding, true),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] must be greater than or equal to 0 if transposed",
             AclarrayToString(params_.outputPadding).c_str()),
             return false);
      if (inputDim == CONV3DINPUTDIM) {
        for (uint64_t i = 0; i < params_.outputPadding->Size(); ++i) {
          OP_CHECK((*params_.outputPadding)[i] < (*params_.stride)[i],
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] should be smaller than stride[%s]",
              AclarrayToString(params_.outputPadding).c_str(), AclarrayToString(params_.stride).c_str()),
              return false);
        }
      }
    } else if (params_.outputPadding != nullptr) { // !transposed, outputPadding value is unneeded
      for (uint64_t i = 0; i < params_.outputPadding->Size(); ++i) {
        OP_CHECK((*params_.outputPadding)[i] == 0,
          OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of outputPadding[%s] must be 0 if not transposed",
            AclarrayToString(params_.outputPadding).c_str()),
            return false);
      }
    }

    // group >= 1
    OP_CHECK(params_.groups >= 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The group[%ld] must be greater than or equal to 1.", params_.groups),
             return false);
    return true;
}

// 需要调整
inline DataType ConvolutionBackwardChecker::CalcPromoteType() {
  auto gradOutputDtype = (inputTensor_.gradOutput)->GetDataType();
  auto inputDtype = (inputTensor_.input)->GetDataType();
  auto weightDtype = (inputTensor_.weight)->GetDataType();

  auto promoteType1 = op::PromoteType(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = op::PromoteType(promoteType1, weightDtype);
  return promoteTypeFinal;
}

bool ConvolutionBackwardChecker::CheckCubeMathTypeConvBackward() {
  auto promoteType = CalcPromoteType();
  return CheckCubeMathType(promoteType, params_.cubeMathType);
}

bool ConvolutionBackwardChecker::CheckConvShape() {
    auto gradOutputDim = inputTensor_.gradOutput->GetViewShape().GetDimNum();
    auto inputDim = inputTensor_.input->GetViewShape().GetDimNum();
    auto weightDim = inputTensor_.weight->GetViewShape().GetDimNum();

    OP_CHECK(gradOutputDim == inputDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of gradOutput and input should be equal."), return false);

    OP_CHECK(inputDim == weightDim, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of input and weight should be equal."),
             return false);
    // 检查gradOutput和weight是否为空tensor
    OP_CHECK(inputTensor_.gradOutput->Size() != 0 && inputTensor_.weight->Size() != 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradOutput and weight cannot be empty tensor."),
             return false);
    if (outputTensor_.gradInput != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor_.input, outputTensor_.gradInput, return false);
    }

    if (outputTensor_.gradWeight != nullptr) {
        OP_CHECK_SHAPE_NOT_EQUAL(inputTensor_.weight, outputTensor_.gradWeight, return false);
    }

    op::Format gradOutputFormat = inputTensor_.gradOutput->GetStorageFormat();
    int64_t channelOutIdx = 0;
    GetChannleIndex(inputTensor_.gradOutput->GetViewShape(), gradOutputFormat, channelOutIdx);
    int64_t cOut = inputTensor_.gradOutput->GetViewShape().GetDim(channelOutIdx);
    if (outputTensor_.gradBias != nullptr) {
        OP_CHECK(outputTensor_.gradBias->GetViewShape().GetDimNum() == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dimension of gradBias only support 1."), return false);
        OP_CHECK(outputTensor_.gradBias->GetViewShape().GetDim(0) == cOut,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The gradBias shape should be equal [%ld].", cOut), return false);
    }

    int64_t paramsDim = inputDim - 2;  // 参数维度应该等于输入tensor维度-2
    int64_t strideDim = params_.stride->Size();
    OP_CHECK(strideDim == paramsDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the input dimension is %ld, the dimension of stride should be %ld.",
                     inputDim, paramsDim),
             return false);

    int64_t dilationDim = params_.dilation->Size();
    OP_CHECK(dilationDim == paramsDim,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "When the input dimension is %ld, the dimension of dilation should be %ld.", inputDim, paramsDim),
             return false);

    int64_t paddingDim = params_.padding->Size();
    int64_t outputPaddingDim = params_.outputPadding->Size();
    if (inputDim == CONV2DINPUTDIM) {
        // padding类支持4维输入
        OP_CHECK(paddingDim == paramsDim || paddingDim == CONV2DINPUTDIM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of padding should be %ld or %d.", inputDim,
                         paramsDim, CONV2DINPUTDIM),
                 return false);

        OP_CHECK(outputPaddingDim == paramsDim || outputPaddingDim == CONV2DINPUTDIM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of outputPadding should be %ld or %d.",
                         inputDim, paramsDim, CONV2DINPUTDIM),
                 return false);
    } else {
        OP_CHECK(
            paddingDim == paramsDim,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the input dimension is %ld, the dimension of padding should be %ld.",
                    inputDim, paramsDim),
            return false);

        OP_CHECK(outputPaddingDim == paramsDim,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "When the input dimension is %ld, the dimension of outputPadding should be %ld.", inputDim,
                         paramsDim),
                 return false);
    }

    return true;
}

bool ConvolutionBackwardChecker::CheckConvChannelAndGroup() {
    op::Shape inputShape = params_.transposed ? inputTensor_.gradOutput->GetViewShape() :
        inputTensor_.input->GetViewShape();
    op::Shape weightShape = inputTensor_.weight->GetViewShape();
    op::Shape gradOutShape = params_.transposed ? inputTensor_.input->GetViewShape() :
        inputTensor_.gradOutput->GetViewShape();
    int64_t inputChannelIdx = 1;
    int64_t gradOutputChannelIdx = 1;
    int64_t weightCoutIdx = 0; // NCDHW & NDHWC
    int64_t weightCinIdx = 1;
    if (socVersion_ == SocVersion::ASCEND910_95) {
      op::Format inputFormat = inputTensor_.input->GetStorageFormat();
      GetChannleIndex(inputShape, inputFormat, inputChannelIdx);
      op::Format gradOutputFormat = inputTensor_.gradOutput->GetStorageFormat();
      GetChannleIndex(gradOutShape, gradOutputFormat, gradOutputChannelIdx);
      op::Format weightFormat = inputTensor_.weight->GetStorageFormat();
      GetChannleIndex(weightShape, weightFormat, weightCinIdx);
    }
    OP_CHECK(gradOutShape.GetDim(gradOutputChannelIdx) == weightShape.GetDim(weightCoutIdx), // 0: NCHW, the order of N(out_channel)
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput_channel(%ld) != weight_N_dim(%ld)",
        gradOutShape.GetDim(gradOutputChannelIdx), weightShape.GetDim(weightCoutIdx)),
        return false);

    bool channelCheck = weightShape.GetDim(weightCinIdx) == 0 ||
        inputShape.GetDim(inputChannelIdx) % weightShape.GetDim(weightCinIdx) != 0;
    OP_CHECK(!channelCheck,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input_channel(%ld) %% weight_channel(%ld) != 0",
        inputShape.GetDim(inputChannelIdx), weightShape.GetDim(weightCinIdx)),
        return false);

    int32_t groups = inputShape.GetDim(inputChannelIdx) / weightShape.GetDim(weightCinIdx);
    bool groupCheck = groups == params_.groups;
    OP_CHECK(groupCheck,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input_channel(%ld) / weight_channel(%ld) != groups(%ld)",
        inputShape.GetDim(inputChannelIdx), weightShape.GetDim(weightCinIdx), params_.groups),
        return false);

    if (inputShape.GetDim(inputChannelIdx) == params_.groups && socVersion_ != SocVersion::ASCEND910_95){
      auto outChannel = gradOutShape.GetDim(inputChannelIdx);
      OP_CHECK(outChannel >= params_.groups,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input_channel(%ld) == groups(%ld), output_channel(%ld) need bigger groups",
        inputShape.GetDim(inputChannelIdx),  params_.groups, outChannel),
        return false);

      OP_CHECK(gradOutShape.GetDim(inputChannelIdx) % params_.groups == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "when input_channel(%ld) == groups(%ld), output_channel(%ld) need k times input_channel",
        inputShape.GetDim(inputChannelIdx),  params_.groups, outChannel),
        return false);
    }

    return true;
}

bool ConvolutionBackwardChecker::CheckConvShapePlus() {
    op::Shape inputShape = params_.transposed ? inputTensor_.gradOutput->GetViewShape() :
        inputTensor_.input->GetViewShape();
    op::Shape weightShape = inputTensor_.weight->GetViewShape();
    op::Shape gradOutShape = params_.transposed ? inputTensor_.input->GetViewShape() :
        inputTensor_.gradOutput->GetViewShape();
    auto inputDim = inputShape.GetDimNum();
    bool expectCheck = false;
    struct ExpectValue expectValue = {};
    int64_t gradOutputCout = 0;
    if (inputDim == CONV3DINPUTDIM) {
        if (socVersion_ != SocVersion::ASCEND910_95) {
          int64_t depthIdx = 2; // NCDHW
          int64_t heightIdx = 3; // NCDHW
          int64_t widthIdx = 4; // NCDHW
          gradOutputCout = gradOutShape.GetDim(1);
          if (!CheckResolutionGEKernelShape(inputShape, weightShape, params_, depthIdx) ||
              !CheckResolutionGEKernelShape(inputShape, weightShape, params_, heightIdx) ||
              !CheckResolutionGEKernelShape(inputShape, weightShape, params_, widthIdx)) {
              return false;
          }
          expectValue.doExpect = GetExpectNum(inputShape, weightShape, params_, depthIdx);
          expectValue.hoExpect = GetExpectNum(inputShape, weightShape, params_, heightIdx);
          expectValue.woExpect = GetExpectNum(inputShape, weightShape, params_, widthIdx);
          expectCheck = expectValue.doExpect == gradOutShape.GetDim(depthIdx) &&
                        expectValue.hoExpect == gradOutShape.GetDim(heightIdx) &&
                        expectValue.woExpect == gradOutShape.GetDim(widthIdx);
          } else {
              OP_CHECK(GetExpectValueDHW_95(inputTensor_, params_, expectValue, inputShape, weightShape),
                      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GetExpectValueDHW_95 failed."), return false);
              int64_t gradOutputDVal = 0;
              int64_t gradOutputHVal = 0;
              int64_t gradOutputWVal = 0;
              op::Format gradOutputFormat = inputTensor_.gradOutput->GetStorageFormat();
              GetInputShapeSize(gradOutputFormat, gradOutShape, gradOutputDVal, gradOutputHVal, gradOutputWVal);
              int64_t coutChannelIdx = 0; // NCDHW
              GetChannleIndex(gradOutShape, gradOutputFormat, coutChannelIdx);
              gradOutputCout = gradOutShape.GetDim(coutChannelIdx);;
              expectCheck = (expectValue.doExpect == gradOutputDVal) && (expectValue.hoExpect == gradOutputHVal) &&
                            (expectValue.woExpect == gradOutputWVal);
          }
          OP_CHECK(expectCheck, OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "gradOutput's shape%s is not equal with inferred shape[%ld,%ld,%ld,%ld,%ld]",
              op::ToString(gradOutShape).GetString(), gradOutShape.GetDim(0), gradOutputCout,
              expectValue.doExpect, expectValue.hoExpect, expectValue.woExpect), return false);
    }
    return true;
}

inline bool ConvolutionBackwardChecker::CheckNotNull() {
    OP_CHECK_NULL(inputTensor_.gradOutput, return false);
    OP_CHECK_NULL(inputTensor_.input, return false);
    OP_CHECK_NULL(inputTensor_.weight, return false);
    OP_CHECK_NULL(params_.stride, return false);
    OP_CHECK_NULL(params_.padding, return false);
    OP_CHECK_NULL(params_.dilation, return false);
    OP_CHECK_NULL(params_.outputPadding, return false);
    OP_CHECK_NULL(params_.outputMask, return false);

    int64_t outputMaskDim = params_.outputMask->Size();
    // outputMask的维度必须为3
    OP_CHECK(outputMaskDim == 3, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim of outputMask must be equal 3."),
             return false);
    if ((*params_.outputMask)[0]) {
        OP_CHECK_NULL(outputTensor_.gradInput, return false);
    }

    if ((*params_.outputMask)[1]) {
        OP_CHECK_NULL(outputTensor_.gradWeight, return false);
    }

    if ((*params_.outputMask)[2]) {
        OP_CHECK_NULL(outputTensor_.gradBias, return false);
    }

    return true;
}

aclnnStatus ConvolutionBackwardChecker::CheckParamsFor8Bit()
{
    if ((*params_.outputMask)[0] && socVersion_ == SocVersion::ASCEND910_95) {
        CHECK_RET(InterceptConvFor8bit(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDtypeValidFor8bit(DataType::DT_HIFLOAT8), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDtypeValidFor8bit(DataType::DT_FLOAT8_E4M3FN), ACLNN_ERR_PARAM_INVALID);
    }
    if ((*params_.outputMask)[1] && socVersion_ == SocVersion::ASCEND910_95) {
        CHECK_RET(InterceptConvFor8bit(), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDtypeValidForBpFilter8bit(DataType::DT_HIFLOAT8), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckParamsValidForBpFilter8bit(), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ConvolutionBackwardChecker::CheckParams() {
    // 检查ConvolutionBackward的输入aclTensor是否符合规范
    //  1. 检查输入aclTensor是否为空指针
    CHECK_RET(CheckNotNull(), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入aclTensor的Dtype是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(inputTensor_.gradOutput, params_.transposed), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(inputTensor_.input, params_.transposed), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(inputTensor_.weight, params_.transposed), ACLNN_ERR_PARAM_INVALID);
    if (outputTensor_.gradInput != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor_.gradInput, params_.transposed), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDataTypeValidForGradInput(), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor_.gradWeight != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor_.gradWeight, params_.transposed), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDataTypeValidForGradWeight(), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor_.gradBias != nullptr) {
        CHECK_RET(CheckDtypeValid(outputTensor_.gradBias, params_.transposed), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDataTypeValidForGradBias(), ACLNN_ERR_PARAM_INVALID);
    }

    // 校验8bit是否支持，支持的情况下，校验8bit输入是否符合条件
    CHECK_RET(CheckParamsFor8Bit() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入aclTensor的Format是否在API支持的数据类型范围之内
    CHECK_RET(CheckFormatValid(inputTensor_.gradOutput, "gradOutput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(inputTensor_.input, "input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(inputTensor_.weight, "weight"), ACLNN_ERR_PARAM_INVALID);
    if (outputTensor_.gradInput != nullptr) {
        CHECK_RET(CheckFormatValid(outputTensor_.gradInput, "gradInput"), ACLNN_ERR_PARAM_INVALID);
    }
    if (outputTensor_.gradWeight != nullptr) {
        CHECK_RET(CheckFormatValid(outputTensor_.gradWeight, "gradWeight"), ACLNN_ERR_PARAM_INVALID);
    }

    if (!params_.transposed && outputTensor_.gradBias != nullptr) {
      if (socVersion_ == SocVersion::ASCEND910B || socVersion_ == SocVersion::ASCEND910_93 || socVersion_ == SocVersion::ASCEND910_95) {
        OP_CHECK(outputTensor_.gradBias->GetStorageFormat() == op::Format::FORMAT_ND,
          OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "gradBias format only support ND, but get [%s].",
            op::ToString(outputTensor_.gradBias->GetStorageFormat()).GetString()), return ACLNN_ERR_PARAM_INVALID);
      }
    }

    // 4. 检查输入参数的合法性
    CHECK_RET(CheckConvParams(inputTensor_.input->GetViewShape().GetDimNum()), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckCubeMathTypeConvBackward(), ACLNN_ERR_PARAM_INVALID);
    if ((inputTensor_.input->Size() == 0 || inputTensor_.weight->Size() == 0) && (socVersion_ == SocVersion::ASCEND910_95 || socVersion_ == SocVersion::ASCEND910B || socVersion_ == SocVersion::ASCEND910_93)) {
      return ACLNN_SUCCESS;
    }
    CHECK_RET(CheckConvShape(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckConvChannelAndGroup(), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckConvShapePlus(), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

bool ConvolutionBackwardChecker::CheckParamsDim() {
  auto inputDim = inputTensor_.input->GetViewShape().GetDimNum();
  uint64_t paramsDim = inputDim - 2; // 参数维度应该等于输入tensor维度-2
  auto validDim = [inputDim, paramsDim](bool condition, const char* paramName) -> bool {
    OP_CHECK(condition,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When the input dimension is %ld, the dimension of %s should be %ld.",
                inputDim,  paramName, paramsDim),
        return false);
    return true;
  };
  bool checkRes =  validDim(params_.stride->Size() == paramsDim, "stride") &&
          validDim(params_.dilation->Size() == paramsDim, "dilation") &&
          validDim(params_.padding->Size() == paramsDim ||
                    (inputDim == CONV2DINPUTDIM && params_.padding->Size() == inputDim), "padding") &&
          validDim(params_.outputPadding->Size() == paramsDim ||
                    (inputDim == CONV2DINPUTDIM && params_.outputPadding->Size() == inputDim), "outputPadding");
  return checkRes;
}

bool ConvolutionBackwardChecker::CheckParamsGroup() {
  op::Shape inputShape = params_.transposed ? inputTensor_.gradOutput->GetViewShape() : inputTensor_.input->GetViewShape();
  op::Shape weightShape = inputTensor_.weight->GetViewShape();
  op::Shape gradOutShape = params_.transposed ? inputTensor_.input->GetViewShape() : inputTensor_.gradOutput->GetViewShape();
  // NCDHW,NCHW,NCL
  auto inputChannel = inputShape.GetDim(1);
  auto weightCin = weightShape.GetDim(1);
  int64_t weightCout = weightShape.GetDim(0);
  OP_CHECK((params_.groups != 0 && weightCout % params_.groups == 0),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output_channel(%ld) %% groups(%ld) != 0",
                weightCout,  params_.groups),
        return false);

  if (params_.transposed && gradOutShape[0] == 0) {
    return true;
  }

  if (inputChannel != 0 || weightCin != 0) {
    bool channelCheck = (weightCin == 0 || inputChannel % weightCin != 0 || inputChannel / weightCin != params_.groups);
    OP_CHECK(!channelCheck,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s(%ld) need groups(%ld) times weight_channel(%ld)",
                params_.transposed ? "gradOutput_channel": "inputChannel",
                inputChannel,  params_.groups, weightCin),
        return false);
  }

  return true;
}

bool ConvolutionBackwardChecker::CheckShape() {
  op::Shape inputShape = inputTensor_.input->GetViewShape();
  op::Shape weightShape = inputTensor_.weight->GetViewShape();
  size_t inputDim = inputShape.GetDimNum();
  // NCDHW,NCHW,NCL
  for (size_t i = 2; i < inputDim; i++) {
    auto index = i - 2; // 2: the dim N and C
    int64_t filterDimDilation = (weightShape[i] - 1) * (*params_.dilation)[index] + 1;
    int64_t dimInput = inputShape.GetDim(i) + (*params_.padding)[index] * 2 - filterDimDilation; // 2: pad two dim
    OP_CHECK(dimInput >= 0,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
      "At dimendion %ld, (in_dim(%ld) + pad_dim(%ld) * 2) should >= ((weight_shape(%ld) - 1) * dilation(%ld) + 1)",
      i, inputShape.GetDim(i), (*params_.padding)[index], weightShape[i], (*params_.dilation)[index]),
      return false);
  }
  return true;
}

bool ConvolutionBackwardChecker::CheckShapeTransposed() {
  int64_t inputC = inputTensor_.input->GetViewShape().GetDim(1);
  int64_t weightCo = inputTensor_.weight->GetViewShape().GetDim(0);
  OP_CHECK(weightCo > 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight Cout should > 0."),
           return false);
  OP_CHECK(weightCo == inputC,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight Cout should be %ld, but get %ld", inputC, weightCo),
           return false);
  return true;
}

bool ConvolutionBackwardChecker::CheckShapeEmpty() {
  if (!CheckParamsDim() || !CheckParamsGroup()) {
    return false;
  }
  if (params_.transposed) {
    return CheckShapeTransposed();
  }
  // 非transposed
  return CheckShape();
}

bool ConvolutionBackwardChecker::CheckEmptyTensor() {
    // 空tensor场景check
    // 空tensor场景且需要计算gradBias, biasSizes不能为nullptr
    if ((*params_.outputMask)[2]) {
        OP_CHECK(params_.biasSizes != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes cannot be nullptr with empty tensor calculation."),
                 return false);
        OP_CHECK(params_.biasSizes->Size() == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes size must be 1, actually is %ld.",
                         params_.biasSizes->Size()),
                 return false);
        int64_t channelOutDim = 1;  // NCHW
        int64_t Cout = inputTensor_.gradOutput->GetViewShape().GetDim(channelOutDim);
        if (!params_.transposed) {
          OP_CHECK((*params_.biasSizes)[0] == Cout,
                   OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The biasSizes should be equal %ld.", Cout), return false);
        } else {
          // transposed=true: bias = weight[Cin] * group
          auto size = inputTensor_.weight->GetViewShape().GetDim(1) * params_.groups;
          OP_CHECK((*params_.biasSizes)[0] == size,
                   OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When transposed=true, the biasSizes should be equal %ld.", size), return false);
        }
    }

    // 空tensor场景只支持input的N和C维度为0
    if ((*params_.outputMask)[0]) {
       auto inputShape = inputTensor_.input->GetViewShape();
      // 确保input的shape大于2
      OP_CHECK(inputShape.GetDimNum() > 2,
              OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The input shape must be greater than 2, but now is %ld",
                      inputShape.GetDimNum()),
              return false);

      OP_CHECK(inputShape[0] == 0 || inputShape[1] == 0,
              OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "ConvolutionBackward only support zero batch or zero channel with input, but got input shape is %s",
                      op::ToString(inputShape).GetString()),
              return false);
    }

    if (socVersion_ == SocVersion::ASCEND910_95 || socVersion_ == SocVersion::ASCEND910B || socVersion_ == SocVersion::ASCEND910_93) {
        return CheckShapeEmpty();
    }
    return true;
}

} // namespace Conv
} // namespace NN
} // namespace Ops
#ifdef __cplusplus
}
#endif
