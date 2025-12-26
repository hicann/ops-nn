/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_convolution_backward.h"
#include "convolutionbackward.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "op_api/op_api_def.h"
#include "../../convolution_forward/op_host/op_api/convolution.h"
#include "pooling/avg_pool3_d_grad/op_host/op_api/dilation.h"
#include "level0/fill.h"
#include "level0/reduce_sum_op.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "level0/zero_op.h"
#include "../../common/op_host/op_api/conv_cube_util.h"
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
#include "convolution_backward_checker.h"
#include "convtbc_backward_checker.h"

using namespace op;
using namespace l0op;
using namespace Ops::NN;
#ifdef __cplusplus
extern "C" {
#endif

const std::vector<DataType> REDUCESUM_SUPPORTED_DTYPES = {
  DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};
static bool IsInputSupportFp32Local() {
  // 判断当前SOC版本是否支持Fp32输入
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
    socVersion == SocVersion::ASCEND910_95) {
    OP_LOGD("The soc version is Ascend910B or ASCEND910_93 or ASCEND910_95, ConvolutionBackward support input tensor with Fp32");
    return true;
  }
  return false;
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

inline DataType CalcPromoteType(const ConvolutionBackwardInputTensor &inputTensor) {
  auto gradOutputDtype = (inputTensor.gradOutput)->GetDataType();
  auto inputDtype = (inputTensor.input)->GetDataType();
  auto weightDtype = (inputTensor.weight)->GetDataType();

  auto promoteType1 = op::PromoteType(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = op::PromoteType(promoteType1, weightDtype);
  return promoteTypeFinal;
}

static bool IsPreInsertDilation(const ConvolutionBackwardInputTensor &inputTensor,
                                const ConvolutionBackwardParams &params) {
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
         ((*params.stride)[kSTRIDEHIdx] > 2 || (*params.stride)[kSTRIDEWIdx] > 2) && !isPreInserDiationSpecialFlag;
}

static bool IsPostInsertDilation(const aclTensor *weight, const ConvolutionBackwardParams &params) {
  return (weight->GetViewShape().GetDim(kHDimNCHWIdx) == 1 && weight->GetViewShape().GetDim(kWDimNCHWIdx) == 1) &&
         ((*params.stride)[kSTRIDEHIdx] > 1 || (*params.stride)[kSTRIDEWIdx] > 1);
}

static bool CheckDeterministic(const int64_t deterministicValue, int groups) {
    OP_CHECK(!((deterministicValue == 1) && (groups > 1)),
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilter cannot support groups(%d) > 1 "
                "in deterministic calculations.", groups),
        return false;
    );
    return true;
}

static bool ConvBackGoHf32(const ConvolutionBackwardInputTensor& inputTensor, int8_t cubeMathType) {
  auto promoteType = op::PromoteType(inputTensor.input->GetDataType(), inputTensor.weight->GetDataType());
  if (inputTensor.gradOutput != nullptr) {
    promoteType = op::PromoteType(promoteType, inputTensor.gradOutput->GetDataType());
  }
  return NeedCubeGoHF32(promoteType, cubeMathType);
}

static inline op::DataType GetUpperFloatDataTypeTransposed(op::DataType a, op::DataType b) {
  OP_LOGD("The input dtype is %s and %s", op::ToString(a).GetString(), op::ToString(b).GetString());
  if (a == op::DataType::DT_FLOAT || b == op::DataType::DT_FLOAT) {
    return op::DataType::DT_FLOAT;
  }
  if (a == op::DataType::DT_BF16 && b == op::DataType::DT_BF16) {
    return op::DataType::DT_BF16;
  }
  if (a == op::DataType::DT_FLOAT16 && b == op::DataType::DT_FLOAT16) {
    return op::DataType::DT_FLOAT16;
  }
  if (a == op::DataType::DT_HIFLOAT8 && b == op::DataType::DT_HIFLOAT8) {
    return op::DataType::DT_HIFLOAT8;
  }
  return op::DataType::DT_FLOAT;  // 注意，原则上a,b都是float类型，若不是，则默认用FP32计算
}

static inline DataType CalcPromoteTypeTransposed(const ConvolutionBackwardInputTensor &inputTensor) {
  auto gradOutputDtype = (inputTensor.gradOutput)->GetDataType();
  auto inputDtype = (inputTensor.input)->GetDataType();
  auto weightDtype = (inputTensor.weight)->GetDataType();

  auto promoteType1 = GetUpperFloatDataTypeTransposed(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = GetUpperFloatDataTypeTransposed(promoteType1, weightDtype);
  return promoteTypeFinal;
}

// DW V2当前支持白名单case
static bool IsC04WhiteListCase(const ConvolutionBackwardInputTensor &inputTensor,
                               const ConvolutionBackwardParams &params) {
  const int64_t expendedDim = 2;
  for (auto i = 0; i < expendedDim; i++) {
    if ((*params.dilation)[i] != 1) {
      return false;
    }
  }
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  return (socVersion == SocVersion::ASCEND910B &&
          (inputTensor.input->GetDataType() == DataType::DT_FLOAT16 ||
           inputTensor.input->GetDataType() == DataType::DT_BF16) &&
          inputTensor.input->GetViewShape().GetDim(0) == 1024 && inputTensor.input->GetViewShape().GetDim(1) == 3 &&
          inputTensor.input->GetViewShape().GetDim(2) == 224 && inputTensor.input->GetViewShape().GetDim(3) == 224 &&
          inputTensor.gradOutput->GetViewShape().GetDim(0) == 1024 &&
          inputTensor.gradOutput->GetViewShape().GetDim(1) == 1024 &&
          inputTensor.gradOutput->GetViewShape().GetDim(2) == 14 &&
          inputTensor.gradOutput->GetViewShape().GetDim(3) == 14 && params.groups == 1);
}

static bool CheckSupportedForConv3dBackpropFilter(ConvolutionBackwardInputTensor &inputTensor,
                                                  ConvolutionBackwardResult &outputTensor,
                                                  ConvolutionBackwardParams &params)
{
  if (!(*params.outputMask)[1] || params.groups == 1) {
    return true;
  }

  op::DataType gradWeightDtype = outputTensor.gradWeight->GetDataType();
  if (gradWeightDtype != DataType::DT_FLOAT) {
    return true;
  }

  if (params.cubeMathType == USE_FP16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is not supported for Conv3DBackpropFilter when promoted input dtype is fp16/bf16, "
      "output dtype is fp32 and group > 1. cubeMathType=%d, please consider to change the cubeMathType to 0, 1 or 3."
      , params.cubeMathType);
    return false;
  }

  auto promoteType = CalcPromoteType(inputTensor);
  if (promoteType == DataType::DT_FLOAT16 || promoteType == DataType::DT_BF16) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "It is not supported for Conv3DBackpropFilter when promoted input dtype is fp16/bf16, "
      "output dtype is fp32 and group > 1. gradOutputDtype=%d, inputDtype=%d, weightDtype=%d, "
      "please consider to change the input dtype to fp32."
      , (inputTensor.gradOutput)->GetDataType(), (inputTensor.input)->GetDataType(), (inputTensor.weight)->GetDataType());
    return false;
  }

  return true;
}

static const aclTensor *Permute(const aclTensor *input, std::vector<int64_t> dims, aclOpExecutor *executor) {
  // contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // Transpose
  auto *perm = executor->AllocIntArray(dims.data(), dims.size());
  CHECK_RET(perm != nullptr, nullptr);

  auto *result = l0op::Transpose(contiguousInput, perm, executor);
  CHECK_RET(result != nullptr, nullptr);

  return result;
}

namespace{
static const aclTensor *View4Das5D(const aclTensor *input, aclOpExecutor *executor)
{
    // NCWH -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, NCDHW_D_DIM, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor *View5Das4D(const aclTensor *input, aclOpExecutor *executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    auto squeezedInput = l0op::SqueezeNd(input, NCDHW_D_DIM, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}
}

static aclIntArray *View2dAs3d(const aclIntArray *intArray, int64_t expendValue, aclOpExecutor *executor) {
  const uint64_t newDimSize = CONV3D_ATTR_DIM;
  int64_t data[newDimSize];
  auto size = intArray->Size();
  if (size != CONV2D_ATTR_DIM) {
    return nullptr;
  }
  data[CONV3D_ATTR_D_IDX] = expendValue;
  data[CONV3D_ATTR_H_IDX] = (*intArray)[0];
  data[CONV3D_ATTR_W_IDX] = (*intArray)[1];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

static aclIntArray *ViewPad4dAs6d(const aclIntArray *intArray, int64_t expendValue, aclOpExecutor *executor) {
  const uint64_t newDimSize = CONV3D_PAD_DIM;
  int64_t data[newDimSize];
  auto size = intArray->Size();
  if (size != CONV2DINPUTDIM) {
    return nullptr;
  }
  data[CONV3D_PAD_HEAD_IDX] = expendValue;
  data[CONV3D_PAD_TAIL_IDX] = expendValue;
  data[CONV3D_PAD_TOP_IDX] = (*intArray)[CONV2D_PAD_TOP_IDX];
  data[CONV3D_PAD_BOTTOM_IDX] = (*intArray)[CONV2D_PAD_BOTTOM_IDX];
  data[CONV3D_PAD_LEFT_IDX] = (*intArray)[CONV2D_PAD_LEFT_IDX];
  data[CONV3D_PAD_RIGHT_IDX] = (*intArray)[CONV2D_PAD_RIGHT_IDX];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  return newArray;
}

static aclnnStatus AttrPreProcess(ConvolutionBackwardParams &params, aclOpExecutor *executor)
{
  if (!params.transposed || GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
    return ACLNN_SUCCESS;
  }
  // ASCEND910_95: Conv2DBP场景下，需要把input tensor维度升到3D，同理attr属性同样要升到3D
  if (params.stride->Size() == CONV2D_ATTR_DIM) {
    params.stride = View2dAs3d(params.stride, 1, executor);
  }
  if (params.dilation->Size() == CONV2D_ATTR_DIM) {
    params.dilation = View2dAs3d(params.dilation, 1, executor);
  }
  if (params.padding->Size() == CONV2D_ATTR_DIM) {
    params.padding = View2dAs3d(params.padding, 0, executor);
  } else if (params.padding->Size() == CONV2DINPUTDIM) {
    params.padding = ViewPad4dAs6d(params.padding, 0, executor);
  }
  OP_CHECK(params.stride != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "stride preprocess failed, get nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);
  OP_CHECK(params.padding != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "padding preprocess failed, get nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);
  OP_CHECK(params.dilation != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "dilation preprocess failed, get nullptr."),
           return ACLNN_ERR_INNER_NULLPTR);

  return ACLNN_SUCCESS;
}

static aclnnStatus InputPreProcess(const aclTensor *&inputTensor, const string &tensorName,
                                   ConvolutionBackwardParams &params, const op::DataType promoteDtype,
                                   aclOpExecutor *executor, bool c04Flag = false, bool transDataFlag = true) {
  // API输入预处理 l0ResultTensor -> l0op::Contiguous -> l0op::Cast -> l0op::TransData -> inputTensor
  inputTensor = l0op::Contiguous(inputTensor, executor);
  OP_CHECK(inputTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, "
          "%s with Contiguous return nullptr.", tensorName.c_str()), return ACLNN_ERR_INNER_NULLPTR);
  bool useFp16Input =
      (params.cubeMathType == USE_FP16 || (!IsInputSupportFp32Local() && params.cubeMathType == ALLOW_FP32_DOWN_PRECISION));
  if (useFp16Input) {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 &&
        (promoteDtype == DataType::DT_HIFLOAT8 || promoteDtype == DataType::DT_FLOAT8_E4M3FN)) {
      OP_LOGW("Cube cannot support dtype: %s when cubeMathType is USE_FP16.",
              op::ToString(inputTensor->GetDataType()).GetString());
    } else {
      OP_LOGD("According to the configuration of cubeMathType, use Fp16 to calculation.");
      inputTensor = l0op::Cast(inputTensor, DataType::DT_FLOAT16, executor);
      OP_CHECK(inputTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, "
               "%s with Cast return nullptr.", tensorName.c_str()), return ACLNN_ERR_INNER_NULLPTR);
    }
  } else {
    inputTensor = l0op::Cast(inputTensor, promoteDtype, executor);
    OP_CHECK(inputTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed,"
             " %s with Cast return nullptr.", tensorName.c_str()), return ACLNN_ERR_INNER_NULLPTR);
  }
  auto inputDim = inputTensor->GetViewShape().GetDimNum();
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
      return ACLNN_SUCCESS;
  }
  if (inputDim == CONV3DINPUTDIM) {
    if (tensorName == "weight") {
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z_3D, params.groups, executor);
    } else {
      if (tensorName == "input" && !transDataFlag) {
          return ACLNN_SUCCESS;
      }
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_NDC1HWC0, params.groups, executor);
    }
  } else {
    if (tensorName == "weight") {
      if (c04Flag) {
        inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z_C04, params.groups, executor);
      } else {
        inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_Z, params.groups, executor);
      }
    } else {
      inputTensor = l0op::TransData(inputTensor, Format::FORMAT_NC1HWC0, params.groups, executor);
    }
  }
  OP_CHECK(inputTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, %s with TransData"
           " return nullptr.", tensorName.c_str()), return ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}
static aclnnStatus OutputPostProcessTransposed(const aclTensor *&outputTensor, const aclTensor *&l0ResultTensor,
                                               const string &tensorName, aclOpExecutor *executor) {
  OP_LOGD("%s's l0func result before cast: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  l0ResultTensor = l0op::Cast(l0ResultTensor, outputTensor->GetDataType(), executor);
  OP_LOGD("%s's l0func result after cast: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  OP_CHECK(l0ResultTensor != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cast %s's l0func result failed, return nullptr.", tensorName.c_str()),
           return ACLNN_ERR_INNER_NULLPTR);
  outputTensor = l0ResultTensor;

  return ACLNN_SUCCESS;
}

static aclnnStatus OutputPostProcess(const aclTensor *&outputTensor, const aclTensor *&l0ResultTensor,
                                     const string &tensorName, const int64_t groups, aclOpExecutor *executor) {
  // API输出后处理 l0ResultTensor -> L0op::Cast(定制) -> L0op::Transdata
  OP_LOGD("%s l0ResultTensor is: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  int64_t storageShapeDimSize = (int64_t) l0ResultTensor->GetStorageShape().GetDimNum();
  bool needSpecialCast = (l0ResultTensor->GetDataType() == op::DataType::DT_FLOAT) &&
    (l0ResultTensor->GetStorageShape().GetDim(storageShapeDimSize - 1) == 16) && (groups > 1);
  // 特殊场景，先cast 再transdata规避
  if (needSpecialCast) {
    l0ResultTensor = l0op::CastOnlyForConvBackward(l0ResultTensor, op::DataType::DT_FLOAT16, executor);
    OP_CHECK(l0ResultTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess fail, l0ResultTensor with cast return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *transTensor = nullptr;
    transTensor = l0op::TransData(l0ResultTensor, GetPrimaryFormat(outputTensor->GetOriginalFormat()), groups, executor);
    OP_CHECK(transTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with TransData return nullptr.",
                    tensorName.c_str()),
            return ACLNN_ERR_INNER_NULLPTR);
    outputTensor = l0op::Cast(transTensor, outputTensor->GetDataType(), executor);
    OP_CHECK(outputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with Cast return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
  } else {
    const aclTensor *transTensor = nullptr;
    transTensor = l0op::TransData(l0ResultTensor, GetPrimaryFormat(outputTensor->GetOriginalFormat()), groups, executor);
    OP_CHECK(transTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with TransData return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("%s outputTensor dtype: %s", tensorName.c_str(), op::ToString(outputTensor->GetDataType()).GetString());
    outputTensor = l0op::Cast(transTensor, outputTensor->GetDataType(), executor);
    OP_CHECK(outputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with Cast return nullptr.",
                     tensorName.c_str()),
             return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("%s outputTensor is: %s", tensorName.c_str(), outputTensor->ToString().GetString());
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus OutputPostProcessWithoutTransdata(const aclTensor *&outputTensor, const aclTensor *&l0ResultTensor,
                                     const string &tensorName, aclOpExecutor *executor) {
  OP_LOGD("Output process without transdata.");
  OP_LOGD("%s l0ResultTensor is: %s", tensorName.c_str(), l0ResultTensor->ToString().GetString());
  outputTensor = l0op::Cast(l0ResultTensor, outputTensor->GetDataType(), executor);
  OP_CHECK(outputTensor != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output postprocess failed, %s with Cast return nullptr.",
                    tensorName.c_str()),
            return ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("%s outputTensor is: %s", tensorName.c_str(), outputTensor->ToString().GetString());
  return ACLNN_SUCCESS;
}

static aclnnStatus OutputViewProcess(ConvolutionBackwardResult &ResultTensor, ConvolutionBackwardOutput &outputTensor,
                                     const aclBoolArray *outputMask, aclOpExecutor *executor) {
  // API进行ViewCopy处理最终输出
  // Index 为 0：ViewCopy gradInput
  if ((*outputMask)[0]) {
    auto result = l0op::ViewCopy(ResultTensor.gradInput, outputTensor.gradInput, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradInput with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }
  // Index 为 1：ViewCopy gradWeight
  if ((*outputMask)[1]) {
    auto result = l0op::ViewCopy(ResultTensor.gradWeight, outputTensor.gradWeight, executor);
    OP_CHECK(
        result != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradWeight with ViewCopy return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
  }
  // Index 为 2：ViewCopy gradBias
  if ((*outputMask)[2]) {
    auto result = l0op::ViewCopy(ResultTensor.gradBias, outputTensor.gradBias, executor);
    OP_CHECK(result != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output viewprocess failed, gradBias with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclIntArray *View1dAs2d(const aclIntArray *intArray, int64_t expendValue, aclOpExecutor *executor,
                               const string &tensorName) {
  const uint64_t newDimSize = 2;
  int64_t data[newDimSize];
  uint64_t arraySize = intArray->Size();
  OP_CHECK(arraySize == 1,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The %s's dimension can only be set to 1 with Conv1D, actually is %ld.",
                   tensorName.c_str(), arraySize),
           return nullptr);

  data[0] = expendValue;
  data[1] = (*intArray)[0];
  aclIntArray *newArray = executor->AllocIntArray(data, newDimSize);
  CHECK_RET(newArray != nullptr, nullptr);
  return newArray;
}

static const aclTensor *View4d(const aclTensor *input, aclOpExecutor *executor, const string &tensorName) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);

  // unsqeeze(2)/unsqueeze(2,3)
  auto inputDim = input->GetViewShape().GetDimNum();
  const int64_t expendedDim = 4;
  int64_t append_dim[expendedDim];
  for (auto i = inputDim - 1; i < expendedDim - 1; i++) {
    append_dim[i - inputDim + 1] = i;
  }
  aclIntArray *dim = executor->AllocIntArray(append_dim, expendedDim - inputDim);
  CHECK_RET(dim != nullptr, nullptr);
  auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, dim, executor);
  OP_CHECK(
      unsqueezedInput != nullptr,
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view4d failed, %s with UnsqueezeNd return nullptr.", tensorName.c_str()),
      return nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
  OP_CHECK(reformatInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view4d failed, %s with ReFormat return nullptr.", tensorName.c_str()),
           return nullptr);

  return reformatInput;
}

static const aclTensor *View3d(const aclTensor *input, aclOpExecutor *executor, const string &tensorName) {
  // input NCL->contigious->unsqueeze(2)->reformat NCHW
  // 非连续转连续contigious
  auto contiguousInput = l0op::Contiguous(input, executor);
  CHECK_RET(contiguousInput != nullptr, nullptr);
  // unsqeeze(2)
  const int64_t append_dim[] = {2};
  aclIntArray *dim = executor->AllocIntArray(append_dim, 1);
  CHECK_RET(dim != nullptr, nullptr);
  auto squeezedInput = l0op::SqueezeNd(contiguousInput, dim, executor);
  OP_CHECK(squeezedInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view3d failed, %s with SqueezeNd return nullptr.", tensorName.c_str()),
           return nullptr);

  // reformat
  auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
  OP_CHECK(reformatInput != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The view3d failed, %s with ReFormat return nullptr.", tensorName.c_str()),
           return nullptr);

  return reformatInput;
}

namespace {
static aclIntArray* View1dAs2dWithGroups(
    const int64_t& groups, const aclIntArray* intArray, const int64_t& expendValue,aclOpExecutor* executor, const string& tensorName)
{
    int64_t data[CONV2D_ATTR_DIM];
    uint64_t arraySize = intArray->Size();
    OP_CHECK(
        arraySize == 1,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The %s's dimension must be set to 1 for Conv1D, but the current value is %ld",
                tensorName.c_str(), arraySize),
                return nullptr);
    if (groups == 1) {
        data[0] = expendValue;
        data[1] = (*intArray)[0];
    } else {
        data[0] = (*intArray)[0];
        data[1] = expendValue;
    }
    aclIntArray* newArray = executor->AllocIntArray(data, CONV2D_ATTR_DIM);
    CHECK_RET(newArray != nullptr, nullptr);
    return newArray;
}

static const aclTensor* View3dWithGroups(
    const int64_t& groups, const aclTensor* input, aclOpExecutor* executor, const string& tensorName)
{
    int64_t squeezeDim;
    if (groups == 1) {
        squeezeDim = kHDimNCHWIdx; // reduce NC1L to NCL
    } else {
        squeezeDim = kWDimNCHWIdx; // reduce NCL1 to NCL
    }
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(contiguousInput, squeezeDim, executor);
    OP_CHECK(
        squeezedInput != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "View3dWithGroups failed: %s return nullptr after SqueezeNd operation.",
                tensorName.c_str()),
                return nullptr);
    auto reformatedInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
    OP_CHECK(
        reformatedInput != nullptr,
        OP_LOGE(
            ACLNN_ERR_INNER_NULLPTR, "View3dWithGroup failed: %s return nullptr after ReFormat operation.",
            tensorName.c_str()),
        return nullptr);
    return reformatedInput;
}

static const aclTensor* View4dWithGroups(
    const int64_t& groups, const aclTensor* input, aclOpExecutor* executor, const string& tensorName)
{
    int64_t unSqueezeDim;
    if (groups == 1) {
        unSqueezeDim = kHDimNCHWIdx; // expand NCL to NC1L
    } else {
        unSqueezeDim = kWDimNCHWIdx; // expand NCL to NCL1
    }
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(contiguousInput, unSqueezeDim, executor);
    OP_CHECK(
        unsqueezedInput != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "View4dWithGroups failed: %s return nullptr after UnsqueezeNd operation.",
                tensorName.c_str()),
                return nullptr);
    auto reformatedInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
    OP_CHECK(
        reformatedInput != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "View4dWithGroups failed: %s return nullptr after ReFormat operation.",
                tensorName.c_str()),
                return nullptr);
    return reformatedInput;
}
}


static const aclTensor *PreDilation(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
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
  if (params.padding->Size() == 4) {
    preDilationPadH = (*params.padding)[kPadding4UpIdx] + (*params.padding)[kPadding4DownIdx];
    preDilationPadW = (*params.padding)[kPadding4LeftIdx] + (*params.padding)[kPadding4RightIdx];
  } else {
    preDilationPadH = 2 * (*params.padding)[kPADDINGUPIdx];
    preDilationPadW = 2 * (*params.padding)[kPADDINGLEFTIdx];
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

static const aclTensor *PostDilation(const aclTensor *dxGradInputNC1HWC0, ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardParams &params, aclOpExecutor *executor) {
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
  if (params.padding->Size() == 4) {
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

static const aclTensor *PerformConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                            ConvolutionBackwardParams &params, aclOpExecutor *executor) {
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

static const aclTensor *CalculateConv2DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                                     ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  const aclTensor *dxGradInputNC1HWC0Res = nullptr;
  if (IsPreInsertDilation(inputTensor, params) && IsInputSupportInsertDilation() &&
      inputTensor.input->GetDataType() != DataType::DT_BF16) {
    const aclTensor *preDilationGradOutputNC1HWC0 = nullptr;
    preDilationGradOutputNC1HWC0 = PreDilation(inputTensor, params, executor);
    int64_t newstrideVector[] = {1, 1};
    aclIntArray *newstride = executor->AllocIntArray(newstrideVector, 2);
    CHECK_RET(newstride != nullptr, nullptr);
    ConvolutionBackwardInputTensor newInputTensor = {preDilationGradOutputNC1HWC0, inputTensor.input,
                                                     inputTensor.weight};
    ConvolutionBackwardParams newparams = {params.biasSizes, newstride,         params.padding,
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
    ConvolutionBackwardInputTensor newInputTensor = {inputTensor.gradOutput, newInput, inputTensor.weight};
    ConvolutionBackwardParams newparams = {params.biasSizes, newstride,         newpadding,
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

static aclnnStatus CalculateBiasGrad(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                     aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  if ((*params.outputMask)[2]) {
    OP_LOGD("Enter bias grad Calculate");
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion == SocVersion::ASCEND910_95) {
        bool biasSupport = std::find(REDUCESUM_SUPPORTED_DTYPES.begin(), REDUCESUM_SUPPORTED_DTYPES.end(),
                                    outputTensor.gradBias->GetDataType()) != REDUCESUM_SUPPORTED_DTYPES.end();
        OP_CHECK(biasSupport,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
              "When outputMask[2] = True, the gradBias current dataType[%s] is not supported. "
              "The supported dataType include: DT_FLOAT16, DT_FLOAT, DT_BF16. Please try set outputMask[2] = False",
              op::ToString(outputTensor.gradBias->GetDataType()).GetString()),
            return ACLNN_ERR_INNER_NULLPTR);
    }
    auto gradContiguous = l0op::Contiguous(inputTensor.gradOutput, executor);
    op::Format gradOutputFormat = inputTensor.gradOutput->GetStorageFormat();
    int64_t reshapeListVal0 = inputTensor.gradOutput->GetViewShape().GetDim(1);
    int64_t reshapeListVal1 = -1;
    if ((socVersion == SocVersion::ASCEND910_95) &&
       (gradOutputFormat == op::Format::FORMAT_NHWC || gradOutputFormat == op::Format::FORMAT_NDHWC)) {
      reshapeListVal0 = -1;
      auto c_idx = inputTensor.gradOutput->GetViewShape().GetDimNum() - 1;
      reshapeListVal1 = inputTensor.gradOutput->GetViewShape().GetDim(c_idx);
    }

    // weight coutchannel始终在第0维，且大小与gradOutput channel一致
    const int64_t reshapeList[] = {inputTensor.gradOutput->GetViewShape().GetDim(0), reshapeListVal0, reshapeListVal1};
    // 3: reshapeList 分配的空间大小
    aclIntArray *preshapeList = executor->AllocIntArray(reshapeList, 3);
    CHECK_RET(preshapeList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradReshape = l0op::Reshape(gradContiguous, preshapeList, executor);
    OP_CHECK(gradReshape != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Reshape with gradOutput failed."),
           return ACLNN_ERR_INNER_NULLPTR);
    auto gradReshapeND = l0op::ReFormat(gradReshape, op::Format::FORMAT_ND);
    OP_CHECK(gradReshapeND != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReFormat with gradOutput failed."),
           return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("gradReshapeND: %s", gradReshapeND->ToString().GetString());
    int64_t reduceAxis = 2; // 2 : reduce_sum的axis参数， 表示在第2维进行reduce求和
    if ((socVersion == SocVersion::ASCEND910_95) &&
        (gradOutputFormat == op::Format::FORMAT_NHWC || gradOutputFormat == op::Format::FORMAT_NDHWC)) {
        reduceAxis = 1;
    }
    const int64_t dim[] = {0, reduceAxis};
    // 2: dim 分配的空间大小
    aclIntArray *pdim = executor->AllocIntArray(dim, 2);
    CHECK_RET(pdim != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradBiasResult = l0op::ReduceSumOp(gradReshapeND, pdim, false, executor);
    OP_CHECK(gradBiasResult != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReduceSumOp with gradOutput failed."),
            return ACLNN_ERR_INNER_NULLPTR);
    OP_LOGD("gradBiasResult: %s", gradBiasResult->ToString().GetString());
    CHECK_RET(
        OutputPostProcess(outputTensor.gradBias, gradBiasResult, "gradBias", params.groups, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static int64_t CalcCountByAxisVec(const op::Shape &dataShape, const vector<int64_t> &axisVec)
{
  int64_t count = 1;
  for (auto axis : axisVec) {
    count *= dataShape[axis];
  }
  return count;
}

static const aclTensor *ViewWithShape(const aclTensor *tensor, const op::Shape &shape, aclOpExecutor *executor)
{
  if (shape == tensor->GetViewShape() && shape == tensor->GetStorageShape()) {
    return tensor;
  }
  return executor->CreateView(tensor, shape, tensor->GetViewOffset());
}

static const aclTensor *ViewWithShapeAndReformatND(const aclTensor *tensor, const std::initializer_list<int64_t> &shape,
  aclOpExecutor *executor)
{
  op::Shape shape2d = op::Shape(shape);
  auto tensor2d = ViewWithShape(tensor, shape2d, executor);
  CHECK_RET(tensor2d != nullptr, nullptr);
  return l0op::ReFormat(tensor2d, op::Format::FORMAT_ND);
}

namespace {
static bool Check2DTransTo1x1DwFlag(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params)
{
  bool c04WhiteFlag = IsC04WhiteListCase(inputTensor, params);
  if(!c04WhiteFlag)
  {
    return false;
  }
  // group=1
  if (params.groups != 1) {
    return false;
  }
  // pad=0
  if ((*params.padding)[0] != 0 || (*params.padding)[1] != 0) {
    return false;
  }
  // dilation=1
  if ((*params.dilation)[0] != 1 || (*params.dilation)[1] != 1) {
      return false;
  }
  // stride=kernel
  op::Shape weightShape = inputTensor.weight->GetViewShape();
  if (weightShape.GetDim(kHDimNCHWIdx) != (*params.stride)[kSTRIDEHIdx] ||
      weightShape.GetDim(kWDimNCHWIdx) != (*params.stride)[kSTRIDEWIdx] ) {
    return false;
  }
  //Hout*Hk=Hin and Wout*Wk=Win
  op::Shape inputShape = params.transposed ? inputTensor.gradOutput->GetViewShape() :
        inputTensor.input->GetViewShape();
  op::Shape gradOutShape = params.transposed ? inputTensor.input->GetViewShape() :
        inputTensor.gradOutput->GetViewShape();
  bool isHEqual = gradOutShape.GetDim(2) * weightShape.GetDim(2) == inputShape.GetDim(2);
  bool isWEqual = gradOutShape.GetDim(3) * weightShape.GetDim(3) == inputShape.GetDim(3);
  if (!isHEqual || !isWEqual) {
    return false;
  }
  return true;
}

static const aclTensor *Conv2DBackpropFilterBy1x1Dw(ConvolutionBackwardInputTensor &inputTensor,
  ConvolutionBackwardParams &params, aclOpExecutor *executor, vector<bool> &conv3DBp2MatmulMask)
{
  int64_t batchIdx = 0; // NCHW
  int64_t channelIdx = 1; // NCHW
  int64_t heightIdx = 2; // NCHW
  int64_t widthIdx = 3; // NCHW
  auto xTensor = params.transposed ? inputTensor.gradOutput : inputTensor.input;
  op::Shape inputShape = xTensor->GetViewShape();
  auto wTensor = inputTensor.weight;
  op::Shape weightShape = wTensor->GetViewShape();
  auto gradOutTensor = params.transposed ? inputTensor.input : inputTensor.gradOutput;
  op::Shape gradOutShape = gradOutTensor->GetViewShape();

  int64_t batchDim = inputShape.GetDim(batchIdx);
  int64_t cOutDim = weightShape.GetDim(batchIdx);
  int64_t cInDim = weightShape.GetDim(channelIdx);
  int64_t hKernelDim = weightShape.GetDim(heightIdx);
  int64_t wKernelDim = weightShape.GetDim(widthIdx);
  int64_t hOutDim = gradOutShape.GetDim(heightIdx);
  int64_t wOutDim = gradOutShape.GetDim(widthIdx);
  // X: reshape from(N, Ci, Hi, Wi) to(N*Ci, Ho, Hk, Wo, Wk)
  auto shape = op::ToShapeVector(xTensor->GetViewShape());
  FVector<int64_t> newShape = {batchDim*cInDim, hOutDim, hKernelDim, wOutDim, wKernelDim};
  aclIntArray* shapeArray = executor->AllocIntArray(newShape.data(), newShape.size());
  CHECK_RET(shapeArray != nullptr, nullptr);
  auto xTensorTmp = l0op::Reshape(xTensor, shapeArray, executor);
  CHECK_RET(xTensorTmp != nullptr, nullptr);
  // X: permute(0, 2, 4, 1, 3)
  FVector<int64_t> newShapeDims = {0, 2, 4, 1, 3};
  auto permAfter = executor->AllocIntArray(newShapeDims.data(), newShapeDims.size());
  CHECK_RET(permAfter != nullptr, nullptr);
  xTensorTmp = l0op::Transpose(xTensorTmp, permAfter, executor);
  CHECK_RET(xTensorTmp != nullptr, nullptr);
  // X: reshape from(N*Ci, Hk, Wk, Ho, Wo) to(N, Ci*Hk*Wk, Ho, Wo)
  auto tmpShape = op::Shape({
    batchDim, cInDim * hKernelDim * wKernelDim, hOutDim, wOutDim
  });
  auto xTensorForDw1x1 = ViewWithShape(xTensorTmp, tmpShape, executor);
  CHECK_RET(xTensorForDw1x1 != nullptr, nullptr);
  if (xTensorForDw1x1->GetStorageFormat() != xTensor->GetStorageFormat()) {
    xTensorForDw1x1 = l0op::ReFormat(xTensorForDw1x1, xTensor->GetStorageFormat());
    CHECK_RET(xTensorForDw1x1 != nullptr, nullptr);
  }
  // W: reshape from(Co, Ci, Hk, Wk) to(Co, Ci*Hk*Wk, 1, 1)
  tmpShape = op::Shape({
    cOutDim, cInDim  * hKernelDim * wKernelDim, 1, 1
  });
  auto wTensorForDw1x1 = ViewWithShape(wTensor, tmpShape, executor);
  CHECK_RET(wTensorForDw1x1 != nullptr, nullptr);
  if (wTensorForDw1x1->GetStorageFormat() != wTensor->GetStorageFormat()) {
    wTensorForDw1x1 = l0op::ReFormat(wTensorForDw1x1, wTensor->GetStorageFormat());
    CHECK_RET(wTensorForDw1x1 != nullptr, nullptr);
  }
  // 新建一个ConvolutionBackwardInputTensor
  ConvolutionBackwardInputTensor inputTensorForDw1x1 = {gradOutTensor, xTensorForDw1x1, wTensorForDw1x1};
  // 新建一个ConvolutionBackwardParams: stride[1, 1] pad[0, 0]
  int64_t newStrideArray[] = {1, 1};
  int64_t newPaddingArray[] = {0, 0};
  aclIntArray *newStride = executor->AllocIntArray(newStrideArray, CONV2D_ATTR_DIM);
  CHECK_RET(newStride != nullptr, nullptr);
  aclIntArray *newPadding = executor->AllocIntArray(newPaddingArray, CONV2D_ATTR_DIM);
  CHECK_RET(newPadding != nullptr, nullptr);
  ConvolutionBackwardParams paramsForDw1x1 = {params.biasSizes, newStride, newPadding,
                                              params.dilation, params.transposed, params.outputPadding,
                                              params.groups, params.outputMask, params.cubeMathType};
  auto promoteType = CalcPromoteType(inputTensorForDw1x1);
  InputPreProcess(inputTensorForDw1x1.gradOutput, "gradOutput", paramsForDw1x1, promoteType, executor);
  InputPreProcess(inputTensorForDw1x1.input, "input", paramsForDw1x1, promoteType, executor);
  InputPreProcess(inputTensorForDw1x1.weight, "weight", paramsForDw1x1, promoteType, executor, false);
  OP_LOGD("inputTensorForDw1x1.input is: %s", inputTensorForDw1x1.input->ToString().GetString());
  OP_LOGD("inputTensorForDw1x1.weight is: %s", inputTensorForDw1x1.weight->ToString().GetString());
  OP_LOGD("inputTensorForDw1x1.gradOutput is: %s", inputTensorForDw1x1.gradOutput->ToString().GetString());
  const aclTensor *gradWeight1x1 = l0op::Conv2DBackpropFilterFp162Fp32(inputTensorForDw1x1.input, inputTensorForDw1x1.weight,
                                                            inputTensorForDw1x1.gradOutput, paramsForDw1x1.stride, paramsForDw1x1.padding,
                                                            paramsForDw1x1.dilation, paramsForDw1x1.groups, executor);
  CHECK_RET(gradWeight1x1 != nullptr, nullptr);
  conv3DBp2MatmulMask[1] = true;
  return gradWeight1x1;
}
static aclnnStatus Conv2DBackpropFilterBy1x1DwReshapeResult(ConvolutionBackwardInputTensor &inputTensor,
  const aclTensor *&outputTensor,aclOpExecutor *executor)
{
  // gradWeight: reshape from(Co, Ci*Hk*Wk, 1, 1) to(Co, Ci, Hk, Wk)
  op::Shape weightShape = inputTensor.weight->GetViewShape();
  int64_t batchIdx = 0; // NCHW
  int64_t channelIdx = 1; // NCHW
  int64_t heightIdx = 2; // NCHW
  int64_t widthIdx = 3; // NCHW
  int64_t cOutDim = weightShape.GetDim(batchIdx);
  int64_t cInDim = weightShape.GetDim(channelIdx);
  int64_t hKernelDim = weightShape.GetDim(heightIdx);
  int64_t wKernelDim = weightShape.GetDim(widthIdx);
  FVector<int64_t> newShape = {cOutDim, cInDim, hKernelDim, wKernelDim};
  auto shapeArray = executor->AllocIntArray(newShape.data(), newShape.size());
  CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
  outputTensor = l0op::Reshape(outputTensor, shapeArray, executor);
  CHECK_RET(outputTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}
}


static aclnnStatus CalculateConv2DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  // 910_95：无2D原型，直接抛错
  OP_CHECK(socVersion != SocVersion::ASCEND910_95,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "No kernel for Conv2DBackwardFiler"),
           return ACLNN_ERR_INNER_NULLPTR);
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  vector<bool> conv2DBp2MatmulMask {false, false};
  if((socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) && (*params.outputMask)[1] && !conv2DBp2MatmulMask[1]) {
    bool transTo1x1DwFlag = false;
    const aclTensor *gradWeightFZ = nullptr;
    transTo1x1DwFlag = Check2DTransTo1x1DwFlag(inputTensor, params);
    if(transTo1x1DwFlag){
      gradWeightFZ = Conv2DBackpropFilterBy1x1Dw(inputTensor, params, executor, conv2DBp2MatmulMask);
      OP_CHECK(gradWeightFZ != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
      CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ, "gradWeight", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
      CHECK_RET(Conv2DBackpropFilterBy1x1DwReshapeResult(inputTensor,outputTensor.gradWeight,executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    }
    if (!(*params.outputMask)[0] && conv2DBp2MatmulMask[1]){
      return ACLNN_SUCCESS;
    }
  }
  auto promoteType = CalcPromoteType(inputTensor);
  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  bool c04Flag = IsC04WhiteListCase(inputTensor, params);
  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor, c04Flag) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());

  // Index 为 0：进行dx运算
  if ((*params.outputMask)[0] && !conv2DBp2MatmulMask[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNC1HWC0 = nullptr;
    gradInputNC1HWC0 = CalculateConv2DBackpropInput(inputTensor, params, executor);
    OP_CHECK(gradInputNC1HWC0 != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropInput return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto status = OutputPostProcess(outputTensor.gradInput, gradInputNC1HWC0, "gradInput", params.groups, executor);
    CHECK_RET(status == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  }

  // Index 为 1：进行dw运算
  if ((*params.outputMask)[1] && !conv2DBp2MatmulMask[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ = nullptr;
    bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
    auto inputDtype = inputTensor.input->GetDataType();

    if (useHf32) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterHf32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                          params.padding, params.dilation, params.groups, executor);
    } else if (inputDtype == DataType::DT_FLOAT) {
      gradWeightFZ = l0op::Conv2DBackpropFilterFp322Fp32(inputTensor.input, inputTensor.weight,
                                                          inputTensor.gradOutput, params.stride, params.padding,
                                                          params.dilation, params.groups, executor);
    } else if (inputDtype == DataType::DT_BF16) {
      gradWeightFZ = l0op::Conv2DBackpropFilterBf162Fp32(inputTensor.input, inputTensor.weight,
                                                          inputTensor.gradOutput, params.stride, params.padding,
                                                          params.dilation, params.groups, executor);
    } else {
      gradWeightFZ = l0op::Conv2DBackpropFilterFp162Fp32(inputTensor.input, inputTensor.weight,
                                                          inputTensor.gradOutput, params.stride, params.padding,
                                                          params.dilation, params.groups, executor);
    }
    OP_CHECK(gradWeightFZ != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ, "gradWeight", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }

  return ACLNN_SUCCESS;
}

static aclnnStatus CalcConv2DBackTransposeInputGrad(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params,
                                                    aclOpExecutor *executor,
                                                    bool useHf32) {
  if (!(*params.outputMask)[0]) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Enter dx Calculate");
  const aclTensor *gradInputTmp = nullptr;
  aclnnStatus outputPostProcessRet = ACLNN_SUCCESS;
  if (useHf32) {
    gradInputTmp = l0op::Conv2d5HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                          params.padding, params.dilation, params.groups, true, executor);
  } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
    gradInputTmp = l0op::Conv2d5HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                          params.padding, params.dilation, params.groups, false, executor);
  } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
    gradInputTmp = l0op::Conv2d5HdBf16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                          params.padding, params.dilation, params.groups, executor);
  } else {
    gradInputTmp = l0op::Conv2d5HdFp16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                          params.padding, params.dilation, params.groups, executor);
  }
  OP_CHECK(gradInputTmp != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with empty tensor failed, Conv2d5Hd return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);

  outputPostProcessRet = OutputPostProcess(outputTensor.gradInput, gradInputTmp, "gradInput",
                                            params.groups, executor);
  CHECK_RET(outputPostProcessRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv2DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // 910_95：无2D原型，直接抛错
  OP_CHECK(GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "No kernel for Conv2DBackwardFiler"),
           return ACLNN_ERR_INNER_NULLPTR);
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

  DataType promoteType =  CalcPromoteType(inputTensor);

  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());
  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  if (CalcConv2DBackTransposeInputGrad(inputTensor, outputTensor, params, executor, useHf32) != ACLNN_SUCCESS) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  if ((*params.outputMask)[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ = nullptr;
    if (useHf32) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterHf32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input, params.stride,
                                        params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterFp322Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                        params.stride, params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterBf162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                        params.stride, params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ =
          l0op::Conv2DBackpropFilterFp162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                      params.stride, params.padding, params.dilation, params.groups, executor);
    }

    OP_CHECK(gradWeightFZ != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ, "gradWeight", params.groups, executor) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

namespace {
static bool SupportConv1DBp2Matmul(const ConvolutionBackwardInputTensor& inputTensor, ConvolutionBackwardParams& params)
{
    // SocVersion must be ASCEND910_95
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
        return false;
    }
    // input format must be NCL
    op::Format format = inputTensor.input->GetStorageFormat();
    if (format != op::Format::FORMAT_NCL) {
        return false;
    }
    // conv1d and matmul don't support 8bit
    auto is8bit = [](const aclTensor* tensor) -> bool {
        auto dtype = tensor->GetDataType();
        return dtype == DataType::DT_HIFLOAT8 || dtype == DataType::DT_FLOAT8_E4M3FN;
    };
    if (is8bit(inputTensor.input) || is8bit(inputTensor.weight) || is8bit(inputTensor.gradOutput))
    {
        return false;
    }
    // transposed must be 0 and groups must be 1
    if (params.transposed || params.groups != 1) {
        return false;
    }
    // padding/outpadding must be 0, and dilation must be 1
    if ((*params.padding)[0] != 0 || (*params.outputPadding)[0] != 0 || (*params.dilation)[0] != 1) {
        return false;
    }
    if (inputTensor.input->GetViewShape()[lDimNCLIdx] == inputTensor.weight->GetViewShape()[lDimNCLIdx] &&
        inputTensor.gradOutput->GetViewShape()[lDimNCLIdx] != 0) {
        return true;
    }
    return false;
}

static aclnnStatus CalculateConv1DBackwardByMatmulImpl(
    ConvolutionBackwardInputTensor& inputTensor, ConvolutionBackwardResult& outputTensor,
    ConvolutionBackwardParams& params, aclOpExecutor* executor)
{
    // The format of 1d is NCL, get n, cin, cout, l
    auto n = inputTensor.input->GetViewShape()[nDimNCLIdx];
    auto inChannels = inputTensor.input->GetViewShape()[cDimNCLIdx];
    auto outChannels = inputTensor.gradOutput->GetViewShape()[cDimNCLIdx];
    auto l = inputTensor.input->GetViewShape()[lDimNCLIdx];
    CalculateBiasGrad(inputTensor, outputTensor, params, executor);
    // Implement Convolution1DBackward dx using matmul
    if ((*params.outputMask)[0]) {
        auto gradOutputND = ViewWithShapeAndReformatND(inputTensor.gradOutput, {n, outChannels}, executor);
        CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto weightND = ViewWithShapeAndReformatND(inputTensor.weight, {outChannels, inChannels * l}, executor);
        CHECK_RET(weightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gradInputND = ViewWithShapeAndReformatND(outputTensor.gradInput, {n, inChannels * l}, executor);
        CHECK_RET(gradInputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto mmGradInput =
            ExecBatchMatmulOp(gradOutputND, weightND, gradInputND, false, false, params.cubeMathType, executor);
        OP_CHECK(mmGradInput != nullptr,
                OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The result of matmul is nullptr in Conv1DBackward prop input"),
                return ACLNN_ERR_INNER_NULLPTR);
        auto gradInput = ViewWithShape(mmGradInput, outputTensor.gradInput->GetViewShape(), executor);
        CHECK_RET(gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto result = l0op::ViewCopy(gradInput, outputTensor.gradInput, executor);
        OP_CHECK(result != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output view process failed, gradInput with ViewCopy in matmul return nullptr"),
                 return ACLNN_ERR_INNER_NULLPTR);
    }

    // 用 matmul 实现 Convolution1DBackward dw
    if ((*params.outputMask)[1]) {
        auto gradOutputND = ViewWithShapeAndReformatND(inputTensor.gradOutput, {n, outChannels}, executor);
        CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputND = ViewWithShapeAndReformatND(inputTensor.input, {n, inChannels * l}, executor);
        CHECK_RET(inputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gradWeightND =
            ViewWithShapeAndReformatND(outputTensor.gradWeight, {outChannels, inChannels * l}, executor);
        CHECK_RET(gradWeightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto mmGradWeight =
            ExecBatchMatmulOp(gradOutputND, inputND, gradWeightND, true, false, params.cubeMathType, executor);
        OP_CHECK(mmGradWeight != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The result of matmul is nullptr in Conv1DBackward prop weight."),
                 return ACLNN_ERR_INNER_NULLPTR);
        auto gradWeight = ViewWithShape(mmGradWeight, outputTensor.gradWeight->GetViewShape(), executor);
        CHECK_RET(gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto result = l0op::ViewCopy(gradWeight, outputTensor.gradWeight, executor);
        OP_CHECK(result != nullptr,
                 OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The output view process failed, gradWeight with ViewCopy in matmul return nullptr."),
                 return ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PreConv1DBackwardTo2D(
    ConvolutionBackwardInputTensor& inputTensor, ConvolutionBackwardResult& outputTensor,
    ConvolutionBackwardParams& params, aclOpExecutor* executor)
{
    params.stride = View1dAs2dWithGroups(params.groups, params.stride, 1, executor, "stride");
    CHECK_RET(params.stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.padding = View1dAs2dWithGroups(params.groups, params.padding, 0, executor, "padding");
    CHECK_RET(params.padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.dilation = View1dAs2dWithGroups(params.groups, params.dilation, 1, executor, "dilation");
    CHECK_RET(params.dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    inputTensor.input = View4dWithGroups(params.groups, inputTensor.input, executor, "input");
    CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);

    inputTensor.weight = View4dWithGroups(params.groups, inputTensor.weight, executor, "weight");
    CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    inputTensor.gradOutput = View4dWithGroups(params.groups, inputTensor.gradOutput, executor, "gradOutput");
    CHECK_RET(inputTensor.gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if ((*params.outputMask)[0]){
      outputTensor.gradInput = View4dWithGroups(params.groups, outputTensor.gradInput, executor, "gradInput");
      CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    
    if ((*params.outputMask)[1]) {
      outputTensor.gradWeight = View4dWithGroups(params.groups, outputTensor.gradWeight, executor, "gradWeight");
      CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PostConv1DBackwardTo2D(
    ConvolutionBackwardResult& outputTensor, ConvolutionBackwardParams& params, aclOpExecutor* executor)
{
    if ((*params.outputMask)[0]) {
        outputTensor.gradInput = View3dWithGroups(params.groups, outputTensor.gradInput, executor, "gradInput");
        CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if ((*params.outputMask)[1]) {
        outputTensor.gradWeight = View3dWithGroups(params.groups, outputTensor.gradWeight, executor, "gradWeight");
        CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}
static aclnnStatus CalculateConv2DBp(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardResult &outputTensor,
                                     ConvolutionBackwardParams &params, aclOpExecutor *executor);
}

static aclnnStatus CalculateConv1DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  if (SupportConv1DBp2Matmul(inputTensor, params)) {
    auto mmStatus = CalculateConv1DBackwardByMatmulImpl(inputTensor, outputTensor, params, executor);
    CHECK_RET(mmStatus == ACLNN_SUCCESS, mmStatus);
  } else {
      auto status = PreConv1DBackwardTo2D(inputTensor, outputTensor, params, executor);
      CHECK_RET(status == ACLNN_SUCCESS, status);
      status = CalculateConv2DBp(inputTensor, outputTensor, params, executor);
      CHECK_RET(status == ACLNN_SUCCESS, status);
      status = PostConv1DBackwardTo2D(outputTensor, params, executor);
      CHECK_RET(status == ACLNN_SUCCESS, status);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv1DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  params.stride = View1dAs2d(params.stride, 1, executor, "stride");
  CHECK_RET(params.stride != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.padding = View1dAs2d(params.padding, 0, executor, "padding");
  CHECK_RET(params.padding != nullptr, ACLNN_ERR_INNER_NULLPTR);

  params.dilation = View1dAs2d(params.dilation, 1, executor, "dilation");
  CHECK_RET(params.dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.input = View4d(inputTensor.input, executor, "input");
  CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.weight = View4d(inputTensor.weight, executor, "weight");
  CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);

  inputTensor.gradOutput = View4d(inputTensor.gradOutput, executor, "gradOutput");
  CHECK_RET(inputTensor.gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if ((*params.outputMask)[0]) {
    outputTensor.gradInput = View4d(outputTensor.gradInput, executor, "gradInput");
    CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if ((*params.outputMask)[1]) {
    outputTensor.gradWeight = View4d(outputTensor.gradWeight, executor, "gradWeight");
    CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  aclnnStatus ret = CalculateConv2DBp(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

  if ((*params.outputMask)[0]) {
    outputTensor.gradInput = View3d(outputTensor.gradInput, executor, "gradInput");
    CHECK_RET(outputTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if ((*params.outputMask)[1]) {
    outputTensor.gradWeight = View3d(outputTensor.gradWeight, executor, "gradWeight");
    CHECK_RET(outputTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static bool IsConv3DBpSupportMatmulMode(const aclTensor *inputTensor, ConvolutionBackwardParams &params)
{
  // must be NCDHW format
  op::Format format = inputTensor->GetStorageFormat();
  if (format != op::Format::FORMAT_NCDHW) {
    return false;
  }
  // padding
  int64_t paddingVecSize = params.padding->Size();
  for (int64_t paddingIdx = 0; paddingIdx < paddingVecSize; ++paddingIdx) {
    if ((*params.padding)[paddingIdx] != 0) {
      return false;
    }
  }
  // outputpadding
  paddingVecSize = params.outputPadding->Size();
  for (int64_t paddingIdx = 0; paddingIdx < paddingVecSize; ++paddingIdx) {
    if ((*params.outputPadding)[paddingIdx] != 0) {
      return false;
    }
  }
  // dilation
  int64_t dilationVecSize = params.dilation->Size();
  for (int64_t dilationIdx = 0; dilationIdx < dilationVecSize; ++dilationIdx) {
    if ((*params.dilation)[dilationIdx] != 1) {
      return false;
    }
  }
  // attribute
  if (params.transposed || params.groups != 1) {
    return false;
  }
  return true;
}

static Conv3DBp2MmMode GetConv3DBp2MmMode(ConvolutionBackwardInputTensor &inputTensor,
  ConvolutionBackwardParams &params)
{
  // matmul暂时不支持8bit
  auto is8bit = [](const aclTensor *tensor) -> bool {
    auto dtype = tensor->GetDataType();
    return dtype == DataType::DT_HIFLOAT8 || dtype == DataType::DT_FLOAT8_E4M3FN;
  };
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 &&
      (is8bit(inputTensor.input) || is8bit(inputTensor.weight) || is8bit(inputTensor.gradOutput))) {
    return Conv3DBp2MmMode::CONV3D_BP_NO_MM;
  }

  const auto &inputShape = inputTensor.input->GetViewShape();
  const auto &weightShape = inputTensor.weight->GetViewShape();
  const auto &gradOutputShape = inputTensor.gradOutput->GetViewShape();
  if (!IsConv3DBpSupportMatmulMode(inputTensor.input, params)) {
    return Conv3DBp2MmMode::CONV3D_BP_NO_MM;
  }

  bool isFmEqKernel = true;
  bool isStrideEqKernel = true;
  const vector<int64_t> dimIdxVec {dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  const vector<int64_t> strideIdxVec {0, 1, 2}; // 0 : depth 1 : height 2 : width
  const int64_t resolutionDimSize = (int64_t)(dimIdxVec.size());
  for (int64_t idx = 0; idx < resolutionDimSize; ++idx) {
    int64_t dimIdx = dimIdxVec[idx];
    int64_t strideIdx = strideIdxVec[idx];
    bool isDimFmEqKernel = (weightShape[dimIdx] == inputShape[dimIdx]) && (gradOutputShape[dimIdx] == 1);
    if (!isDimFmEqKernel) {
      isFmEqKernel = false;
    }
    bool shapeCondition = weightShape[dimIdx] != 0 && inputShape[dimIdx] % weightShape[dimIdx] == 0
                          && inputShape[dimIdx] / weightShape[dimIdx] == gradOutputShape[dimIdx];
    if (weightShape[dimIdx] != (*params.stride)[strideIdx] || !shapeCondition) {
      isStrideEqKernel = false;
    }
  }
  if (isFmEqKernel) {
    return Conv3DBp2MmMode::CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL;
  }
  // kernel=1*1或2*2走拆分性能略好于matmul
  bool notKernelSplit = (*params.stride)[1] != (*params.stride)[2] || (*params.stride)[1] > 2;
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 && isStrideEqKernel && notKernelSplit) {
    return Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL;
  }
  return Conv3DBp2MmMode::CONV3D_BP_NO_MM;
}

static aclnnStatus GenDxInOutByConvBp2MmMode(BatchMatmulInput &batchMmInput,
                                             ConvolutionBackwardInputTensor &inputTensor,
                                             ConvolutionBackwardResult &outputTensor,
                                             aclOpExecutor *executor,
                                             Conv3DBp2MmMode conv2MmMode)
{
  auto gradOutput = inputTensor.gradOutput;
  auto weight = inputTensor.weight;
  auto gradInput = outputTensor.gradInput;
  vector<int64_t> gradShapeVec {0, 0, 0};
  const int64_t bDimBMNIdx = 0; // batch in [batch, m, n] for mamtul
  const int64_t mDimBMNIdx = 1; // m in [batch, m, n] for mamtul
  const int64_t nDimBMNIdx = 2; // n in [batch, m, n] for mamtul
  const vector<int64_t> dhwIdxUnionVec {dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  const vector<int64_t> cidhwIdxUnionVec {ciDimCoCiDHWIdx, dDimNCDHWIdx, hDimNCDHWIdx, wDimNCDHWIdx};
  // input : gradOutput
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ||
    conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    gradShapeVec = {gradOutput->GetViewShape()[nDimNCDHWIdx], gradOutput->GetViewShape()[cDimNCDHWIdx],
      CalcCountByAxisVec(gradOutput->GetViewShape(), dhwIdxUnionVec)};
  } else if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL) {
    gradShapeVec = {gradOutput->GetViewShape()[nDimNCDHWIdx], 1, gradOutput->GetViewShape()[1]};
  }
  auto gradOutputND = ViewWithShapeAndReformatND(gradOutput,
    {gradShapeVec[bDimBMNIdx], gradShapeVec[mDimBMNIdx], gradShapeVec[nDimBMNIdx]}, executor);
  CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // input : weight -- 1, Co, CinDHW
  std::initializer_list<int64_t> weightShape {1, weight->GetViewShape()[coDimCoCiDHWIdx],
    CalcCountByAxisVec(weight->GetViewShape(), cidhwIdxUnionVec)};
  auto weightND = ViewWithShapeAndReformatND(weight, weightShape, executor);
  CHECK_RET(weightND != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // output
  vector<int64_t> outShapeVec {0, 0, 0};
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    // CONV3D_BP_MM_STRIDE_EQ_KERNEL's matmul output is N, DoHoWo, CinDkHkWk
    outShapeVec = {gradInput->GetViewShape()[nDimNCDHWIdx],
      CalcCountByAxisVec(gradOutput->GetViewShape(), dhwIdxUnionVec),
      CalcCountByAxisVec(weight->GetViewShape(), cidhwIdxUnionVec)};
  } else {
    outShapeVec = {gradInput->GetViewShape()[nDimNCDHWIdx],
      gradInput->GetViewShape()[cDimNCDHWIdx],
      CalcCountByAxisVec(gradInput->GetViewShape(), dhwIdxUnionVec)};
  }
  auto mmDxOutputND = ViewWithShapeAndReformatND(gradInput,
    {outShapeVec[bDimBMNIdx], outShapeVec[mDimBMNIdx], outShapeVec[nDimBMNIdx]}, executor);
  CHECK_RET(mmDxOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);

  batchMmInput.leftData = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ? weightND : gradOutputND;
  batchMmInput.isLeftTranspose = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ||
    conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL;
  batchMmInput.rightData = conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_1x1_KERNEL ? gradOutputND : weightND;
  batchMmInput.isRightTranspose = false;
  batchMmInput.outputData = mmDxOutputND;
  return ACLNN_SUCCESS;
}

static const aclTensor *DoPostMatmulForConv3dBpInput(const aclTensor *gradInputND,
  ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardResult &outputTensor,
  aclOpExecutor *executor, Conv3DBp2MmMode conv2MmMode)
{
  // permute : to recover data arrangement : NDoHoWoCinDkHkWk -> NCinDoDkHoHkWoWk
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_STRIDE_EQ_KERNEL) {
    const auto &gradOutputShape = inputTensor.gradOutput->GetViewShape();
    const auto &weightShape = inputTensor.weight->GetViewShape();
    op::Shape mmOutputExtendShape = op::Shape({
      gradOutputShape[0], gradOutputShape[dDimNCDHWIdx], gradOutputShape[hDimNCDHWIdx], gradOutputShape[wDimNCDHWIdx],
      weightShape[ciDimCoCiDHWIdx], weightShape[dDimNCDHWIdx], weightShape[hDimNCDHWIdx], weightShape[wDimNCDHWIdx],
    });
    auto mmOutputExtendND = ViewWithShape(gradInputND, mmOutputExtendShape, executor);
    CHECK_RET(mmOutputExtendND != nullptr, nullptr);
    std::vector<int64_t> permDimVec {0, 4, 1, 5, 2, 6, 3, 7}; // NDoHoWoCinDkHkWk -> NCinDoDkHoHkWoWk
    gradInputND = Permute(mmOutputExtendND, permDimVec, executor);
  }
  // ND -> NCDHW
  CHECK_RET(gradInputND != nullptr, nullptr);
  auto gradInputNCDHW = ViewWithShape(gradInputND, outputTensor.gradInput->GetViewShape(), executor);
  CHECK_RET(gradInputNCDHW != nullptr, nullptr);
  gradInputNCDHW = l0op::ReFormat(gradInputNCDHW, op::Format::FORMAT_NCDHW);
  return gradInputNCDHW;
}

// FM=KERNEL
static aclnnStatus GenConvMmDwInputByMode(BatchMatmulInput &batchMmInput,
                                       ConvolutionBackwardInputTensor &inputTensor,
                                       aclOpExecutor *executor)
{
  auto gradOutput = inputTensor.gradOutput; // dy
  auto input = inputTensor.input; //x
  op::Shape gradOutputShape2d = op::Shape({
    1,
    gradOutput->GetViewShape()[0],
    CalcCountByAxisVec(gradOutput->GetViewShape(), {1, 2, 3, 4})});
  auto gradOutput2d = ViewWithShape(gradOutput, gradOutputShape2d, executor);
  CHECK_RET(gradOutput2d != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradOutputND = l0op::ReFormat(gradOutput2d, op::Format::FORMAT_ND);

  op::Shape inputShape2d = op::Shape({
      1,
      input->GetViewShape()[0],
      CalcCountByAxisVec(input->GetViewShape(), {1, 2, 3, 4})});
  auto input2d = ViewWithShape(input, inputShape2d, executor);
  CHECK_RET(input2d != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto inputND = l0op::ReFormat(input2d, op::Format::FORMAT_ND);

  batchMmInput.leftData = gradOutputND;
  batchMmInput.isLeftTranspose = true;
  batchMmInput.rightData = inputND;
  batchMmInput.isRightTranspose = false;
  return ACLNN_SUCCESS;
}

static aclnnStatus GenConvMmDwOutputByMode(aclTensor *&mmDwOutputNDFp32,
                                       ConvolutionBackwardResult &outputTensor,
                                       aclOpExecutor *executor,
                                       [[maybe_unused]] Conv3DBp2MmMode conv2MmMode)
{
    auto gradInput = outputTensor.gradInput;
    auto gradWeight = outputTensor.gradWeight;
    op::Shape mmDwOutShape2d = op::Shape({
      1,
      gradWeight->GetViewShape()[0],  //cout
      CalcCountByAxisVec(gradInput->GetViewShape(), {1, 2, 3, 4})});
    auto mmDwOutput2d = ViewWithShape(gradInput, mmDwOutShape2d, executor);
    CHECK_RET(mmDwOutput2d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto mmDwOutputND = l0op::ReFormat(mmDwOutput2d, op::Format::FORMAT_ND);
    mmDwOutputNDFp32 = executor->CreateView(mmDwOutputND, mmDwOutputND->GetViewShape(),
      mmDwOutputND->GetViewOffset());
    CHECK_RET(mmDwOutputNDFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    mmDwOutputNDFp32->SetDataType(DataType::DT_FLOAT);

  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DBackwardDwByMmMode(ConvolutionBackwardInputTensor &inputTensor,
                                                   ConvolutionBackwardResult &outputTensor,
                                                   ConvolutionBackwardParams &params,
                                                   aclOpExecutor *executor,
                                                   Conv3DBp2MmMode conv2MmMode)
{
  BatchMatmulInput batchMmInput;
  auto status = GenConvMmDwInputByMode(batchMmInput, inputTensor, executor);
  if (status != ACLNN_SUCCESS) {
    return status;
  }
  OP_LOGD("Enter backprop filter Calculate with matmul mode");
  aclTensor *mmDwOutputNDFp32 = nullptr;
  status = GenConvMmDwOutputByMode(mmDwOutputNDFp32, outputTensor, executor, conv2MmMode);
  if (status != ACLNN_SUCCESS) {
    OP_LOGD("GenConvMmDwOutputByMode False");
    return status;
  }
  auto gradWeightNND = ExecBatchMatmulOp(batchMmInput.leftData, batchMmInput.rightData, mmDwOutputNDFp32, batchMmInput.isLeftTranspose,
    batchMmInput.isRightTranspose, params.cubeMathType, executor);
  OP_CHECK(gradWeightNND != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ExecBatchMatmulOp for 3ddw return nullptr."),
            return ACLNN_ERR_INNER_NULLPTR);

  auto gradWeightNCDHW = ViewWithShape(gradWeightNND, outputTensor.gradWeight->GetViewShape(), executor);
  CHECK_RET(gradWeightNCDHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
  gradWeightNCDHW = l0op::ReFormat(gradWeightNCDHW, op::Format::FORMAT_NCDHW);

  OP_LOGD("gradWeightNCDHW: %s", gradWeightNCDHW->ToString().GetString());
  CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightNCDHW, "gradWeight", params.groups, executor) ==
            ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static bool IsW1B1FmNDxTransToMm(const ConvolutionBackwardInputTensor &inputTensor,
                      const ConvolutionBackwardParams &params)
{
  OP_LOGD("Enter IsW1B1FmNDxTransToMm");
  // only support  ASCEND910_95
  if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_95) {
    return false;
  }
  // do not support 8bit
  if (inputTensor.input->GetDataType() == DataType::DT_HIFLOAT8 ||
      inputTensor.input->GetDataType() == DataType::DT_FLOAT8_E4M3FN) {
    return false;
  }
  // only support NCDHW
  if (inputTensor.gradOutput->GetStorageFormat() != op::Format::FORMAT_NCDHW ||
      inputTensor.input->GetStorageFormat() != op::Format::FORMAT_NCDHW ||
      inputTensor.weight->GetStorageFormat() != op::Format::FORMAT_NCDHW) {
    return false;
  }
  // group=1
  if (params.groups != 1) {
    return false;
  }
  // pad=0
  if ((*params.padding)[CONV3D_ATTR_D_IDX] != 0 || (*params.padding)[CONV3D_ATTR_H_IDX] != 0 ||
      (*params.padding)[CONV3D_ATTR_W_IDX] != 0) {
    return false;
  }
  // dilation=1
  if ((*params.dilation)[CONV3D_ATTR_D_IDX] != 1 || (*params.dilation)[CONV3D_ATTR_H_IDX] != 1 ||
      (*params.dilation)[CONV3D_ATTR_W_IDX] != 1) {
      return false;
  }
  // stride=1
  if ((*params.stride)[CONV3D_ATTR_D_IDX] != 1 || (*params.stride)[CONV3D_ATTR_H_IDX] != 1 ||
      (*params.stride)[CONV3D_ATTR_W_IDX] != 1) {
      return false;
  }
  // kernel=1
  op::Shape weightShape = inputTensor.weight->GetViewShape();
  if (weightShape.GetDim(NCDHW_D_DIM) != 1 || weightShape.GetDim(NCDHW_H_DIM) != 1 ||
      weightShape.GetDim(NCDHW_W_DIM) != 1) {
    return false;
  }

  op::Shape inputShape = params.transposed ? inputTensor.gradOutput->GetViewShape() : inputTensor.input->GetViewShape();
  // 本规格仅限batch = 1
  if (inputShape.GetDim(NCDHW_N_DIM) != 1) {
      return false;
  }
  // Din, Hin, Win不能同时为1
  if (inputShape.GetDim(NCDHW_D_DIM) == 1 && inputShape.GetDim(NCDHW_H_DIM) == 1 &&
      inputShape.GetDim(NCDHW_W_DIM) == 1) {
      return false;
  }
  OP_LOGD("Original ConvBackpropInput can convert to Matmul, "
          "from [w.shape(Cout, Cin, 1, 1, 1), Dy.shape(1, Cout, Dout=Din, Hout=Hin, Wout=Win), Dx.shape(1, Cin, Din, Hin, Win)] "
          "to [a.shape(Cout, Cin), b.shape(Cout, Din*Hin*Win), c.shape(Cin, Din*Win*Hin)].");
  return true;
}

static aclnnStatus CalculateW1B1FmNDxByMm(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardResult &outputTensor,
                                          ConvolutionBackwardParams &params, aclOpExecutor *executor)
{
  OP_LOGD("Enter CalculateW1B1FmNDxByMm");
  auto wTensor = inputTensor.weight;
  op::Shape weightShape = wTensor->GetViewShape();
  auto gradOutTensor = params.transposed ? inputTensor.input : inputTensor.gradOutput;
  op::Shape gradOutShape = gradOutTensor->GetViewShape();
  OP_LOGD("wTensor is: %s", wTensor->ToString().GetString());
  OP_LOGD("gradOutTensor is: %s", gradOutTensor->ToString().GetString());
  int64_t cOutDim = weightShape.GetDim(NCDHW_N_DIM);
  int64_t cInDim = weightShape.GetDim(NCDHW_C_DIM);
  int64_t dOutDim = gradOutShape.GetDim(NCDHW_D_DIM);
  int64_t hOutDim = gradOutShape.GetDim(NCDHW_H_DIM);
  int64_t wOutDim = gradOutShape.GetDim(NCDHW_W_DIM);
  // Weight: reshape from(Cout, Cin, 1, 1, 1) to(Cout, Cin)
  op::Shape tmpShape = op::Shape({cOutDim, cInDim});
  auto mat1ForMm = executor->CreateView(wTensor, tmpShape, wTensor->GetViewOffset());
  OP_CHECK_NULL(mat1ForMm, return ACLNN_ERR_INNER_NULLPTR);
  op::Strides mat1ViewStride = op::Strides({mat1ForMm->GetViewStrides()[1], mat1ForMm->GetViewStrides()[0]});
  mat1ForMm->SetStorageFormat(Format::FORMAT_ND);
  mat1ForMm->SetOriginalFormat(Format::FORMAT_ND);
  tmpShape = op::Shape({cInDim, cOutDim});
  mat1ForMm->SetViewShape(tmpShape);
  mat1ForMm->SetViewFormat(Format::FORMAT_ND);
  mat1ForMm->SetViewStrides(mat1ViewStride);
  OP_LOGD("mat1ForMm is: %s", mat1ForMm->ToString().GetString());

  // Dy: reshape from(1, Cout, Dout, Hout, Wout) to(Cout, Dout*Hout*Wout)
  FVector<int64_t> mat2ShapeVec = {cOutDim, dOutDim * hOutDim * wOutDim};
  aclIntArray *mat2ShapeArray = executor->AllocIntArray(mat2ShapeVec.data(), mat2ShapeVec.size());
  OP_CHECK_NULL(mat2ShapeArray, return ACLNN_ERR_INNER_NULLPTR);
  auto mat2ForMm = l0op::Reshape(gradOutTensor, mat2ShapeArray, executor);
  OP_CHECK_NULL(mat2ForMm, return ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("mat2ForMm is: %s", mat2ForMm->ToString().GetString());

  auto matmulOut = ExecMmOp(mat1ForMm, mat2ForMm, params.cubeMathType, executor);
  OP_CHECK_NULL(matmulOut, return ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("matmulOut is: %s", matmulOut->ToString().GetString());
  // Dx: reshape from(Cin, Din*Win*Hin) to(1, Cin, Din=Dout, Hin=Hout, Win=Wout)
  tmpShape = op::Shape({1, cInDim, dOutDim, hOutDim, wOutDim});
  auto *gradInputND = ViewWithShape(matmulOut, tmpShape, executor);
  OP_CHECK_NULL(gradInputND, return ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("gradInputND is: %s", gradInputND->ToString().GetString());
  auto status = OutputPostProcessWithoutTransdata(outputTensor.gradInput, gradInputND, "gradInput", executor);
  CHECK_RET(status == ACLNN_SUCCESS, status);
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DBackwardByMatmulImpl(ConvolutionBackwardInputTensor &inputTensor,
                                                       ConvolutionBackwardResult &outputTensor,
                                                       ConvolutionBackwardParams &params,
                                                       aclOpExecutor *executor,
                                                       vector<bool> &conv3DBp2MatmulMask)
{
  OP_LOGD("Enter CalculateConv3DBackwardByMatmulImpl");
  // 是否满足新的条件
  bool w1B1FmNDxTransToMmFlag = IsW1B1FmNDxTransToMm(inputTensor, params); // 判定条件函数
  auto conv2MmMode = GetConv3DBp2MmMode(inputTensor, params);
  if (conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_NO_MM && !w1B1FmNDxTransToMmFlag) {
    return ACLNN_SUCCESS;
  }
  if ((*params.outputMask)[0]) {
    if (w1B1FmNDxTransToMmFlag) {
      // 新的adapt逻辑
      auto status = CalculateW1B1FmNDxByMm(inputTensor, outputTensor, params,executor);
      CHECK_RET(status == ACLNN_SUCCESS, status);
      conv3DBp2MatmulMask[0] = true;
    } else {
      BatchMatmulInput batchMmInput;
      auto status = GenDxInOutByConvBp2MmMode(batchMmInput, inputTensor, outputTensor, executor, conv2MmMode);
      CHECK_RET(status == ACLNN_SUCCESS, status);
      OP_LOGD("Enter Conv3DBackpropInput Calculation By Matmul Implementation.");
      auto gradInputND = ExecBatchMatmulOp(batchMmInput.leftData, batchMmInput.rightData, batchMmInput.outputData,
        batchMmInput.isLeftTranspose, batchMmInput.isRightTranspose, params.cubeMathType, executor);
      OP_CHECK(
          gradInputND != nullptr,
          OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Mamtul In Conv3DBackpropInput Return Nullptr."),
          return ACLNN_ERR_INNER_NULLPTR);
      auto gradInputNCDHW = DoPostMatmulForConv3dBpInput(gradInputND, inputTensor, outputTensor, executor, conv2MmMode);
      CHECK_RET(gradInputNCDHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
      status = OutputPostProcess(outputTensor.gradInput, gradInputNCDHW, "gradInput", params.groups, executor);
      CHECK_RET(status == ACLNN_SUCCESS, status);
      conv3DBp2MatmulMask[0] = true;
    }
  }
  // 当前dw的实现仅支持FM等于Kernel场景
  if ((*params.outputMask)[1] && conv2MmMode == Conv3DBp2MmMode::CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL) {
    auto status = CalculateConv3DBackwardDwByMmMode(inputTensor, outputTensor, params, executor, conv2MmMode);
    CHECK_RET(status == ACLNN_SUCCESS, status);
    conv3DBp2MatmulMask[1] = true;
  }
  return ACLNN_SUCCESS;
}
namespace {
static const aclTensor *GetGradInputNDC1HWC0(const ConvolutionBackwardInputTensor &inputTensor,
                                       const ConvolutionBackwardParams &params,
                                       aclOpExecutor *executor, bool useHf32)
{
    const aclTensor *gradInputNDC1HWC0 = nullptr;
    if (useHf32) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputHf32(inputTensor.input, inputTensor.weight,
                                                    inputTensor.gradOutput, params.stride,
                                                    params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputFp322Fp32(inputTensor.input, inputTensor.weight,
                                                            inputTensor.gradOutput, params.stride,
                                                            params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputBf162Bf16(inputTensor.input, inputTensor.weight,
                                                            inputTensor.gradOutput, params.stride,
                                                            params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInputFp162Fp16(inputTensor.input, inputTensor.weight,
                                                            inputTensor.gradOutput, params.stride,
                                                            params.padding, params.dilation, params.groups, executor);
    }
    return gradInputNDC1HWC0;
}

static const aclTensor *GetGradWeightFZ3D(const ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
                                    aclOpExecutor *executor, bool useHf32)
{
    const aclTensor *gradWeightFZ3D = nullptr;
    if (useHf32) {
      gradWeightFZ3D = l0op::Conv3DBackpropFilterHf32(inputTensor.input, inputTensor.weight,
                                            inputTensor.gradOutput, params.stride,
                                            params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp322Fp32(inputTensor.input, inputTensor.weight,
                                            inputTensor.gradOutput, params.stride,
                                            params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterBf162Fp32(inputTensor.input, inputTensor.weight,
                                            inputTensor.gradOutput, params.stride,
                                            params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp162Fp32(inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
                                        params.padding, params.dilation, params.groups, executor);
    }
    return gradWeightFZ3D;
}

bool IsTransTo1x1Dw(const ConvolutionBackwardInputTensor &inputTensor,
                  const ConvolutionBackwardParams &params)
{
  OP_LOGD("Enter IsTransTo1x1Dw");
  // do not support 8bit
  if (inputTensor.input->GetDataType() == DataType::DT_HIFLOAT8 ||
      inputTensor.input->GetDataType() == DataType::DT_FLOAT8_E4M3FN) {
    return false;
  }
  // only support NCDHW
  if (inputTensor.gradOutput->GetStorageFormat() != op::Format::FORMAT_NCDHW ||
      inputTensor.input->GetStorageFormat() != op::Format::FORMAT_NCDHW ||
      inputTensor.weight->GetStorageFormat() != op::Format::FORMAT_NCDHW) {
    return false;
  }
  // group=1
  if (params.groups != 1) {
    return false;
  }
  // pad=0
  if ((*params.padding)[CONV3D_ATTR_D_IDX] != 0 || (*params.padding)[CONV3D_ATTR_H_IDX] != 0 ||
      (*params.padding)[CONV3D_ATTR_W_IDX] != 0) {
    return false;
  }
  // dilation=1
  if ((*params.dilation)[CONV3D_ATTR_D_IDX] != 1 || (*params.dilation)[CONV3D_ATTR_H_IDX] != 1 ||
      (*params.dilation)[CONV3D_ATTR_W_IDX] != 1) {
      return false;
  }
  // stride=kernel
  op::Shape weightShape = inputTensor.weight->GetViewShape();
  if (weightShape.GetDim(NCDHW_D_DIM) != (*params.stride)[CONV3D_ATTR_D_IDX] ||
      weightShape.GetDim(NCDHW_H_DIM) != (*params.stride)[CONV3D_ATTR_H_IDX] ||
      weightShape.GetDim(NCDHW_W_DIM) != (*params.stride)[CONV3D_ATTR_W_IDX]) {
    return false;
  }
  // Dout*Dk=Din, Hout*Hk=Hin and Wout*Wk=Win
  op::Shape inputShape = params.transposed ? inputTensor.gradOutput->GetViewShape() : inputTensor.input->GetViewShape();
  op::Shape gradOutShape = params.transposed ? inputTensor.input->GetViewShape() : inputTensor.gradOutput->GetViewShape();
  bool isDEqual = gradOutShape.GetDim(NCDHW_D_DIM) * weightShape.GetDim(NCDHW_D_DIM) == inputShape.GetDim(NCDHW_D_DIM);
  bool isHEqual = gradOutShape.GetDim(NCDHW_H_DIM) * weightShape.GetDim(NCDHW_H_DIM) == inputShape.GetDim(NCDHW_H_DIM);
  bool isWEqual = gradOutShape.GetDim(NCDHW_W_DIM) * weightShape.GetDim(NCDHW_W_DIM) == inputShape.GetDim(NCDHW_W_DIM);
  if (!isDEqual || !isHEqual || !isWEqual) {
    return false;
  }
  // 性能判定条件：需要 hk >= 8 and wk >= 8
  if ((*params.stride)[CONV3D_ATTR_H_IDX] < 8 || (*params.stride)[CONV3D_ATTR_W_IDX] < 8) { // 8: 经验值，stride越大Dw搬运膨胀越大，越容易回本
    return false;
  }

  OP_LOGD("Original ConvBackpropFilter can convert to ConvBackpropFilter with kernelD=kernelH=kernelW=1.");
  return true;
}

static const aclTensor *Conv3DBackpropFilterBy1x1Dw(ConvolutionBackwardInputTensor &inputTensor,
  ConvolutionBackwardParams &params, aclOpExecutor *executor, bool useHf32)
{
  OP_LOGD("Enter Conv3DBackpropFilterBy1x1Dw");
  auto xTensor = params.transposed ? inputTensor.gradOutput : inputTensor.input;
  op::Shape inputShape = xTensor->GetViewShape();
  auto wTensor = inputTensor.weight;
  op::Shape weightShape = wTensor->GetViewShape();
  auto gradOutTensor = params.transposed ? inputTensor.input : inputTensor.gradOutput;
  op::Shape gradOutShape = gradOutTensor->GetViewShape();
  OP_LOGD("xTensor is: %s", xTensor->ToString().GetString());
  OP_LOGD("wTensor is: %s", wTensor->ToString().GetString());
  OP_LOGD("gradOutTensor is: %s", gradOutTensor->ToString().GetString());
  int64_t batchDim = inputShape.GetDim(NCDHW_N_DIM);
  int64_t cOutDim = weightShape.GetDim(NCDHW_N_DIM);
  int64_t cInDim = weightShape.GetDim(NCDHW_C_DIM);
  int64_t dKernelDim = weightShape.GetDim(NCDHW_D_DIM);
  int64_t hKernelDim = weightShape.GetDim(NCDHW_H_DIM);
  int64_t wKernelDim = weightShape.GetDim(NCDHW_W_DIM);
  int64_t dOutDim = gradOutShape.GetDim(NCDHW_D_DIM);
  int64_t hOutDim = gradOutShape.GetDim(NCDHW_H_DIM);
  int64_t wOutDim = gradOutShape.GetDim(NCDHW_W_DIM);
  // X: reshape from(N, Ci, Di, Hi, Wi) to(N*Ci, Do, Dk, Ho, Hk, Wo, Wk)
  FVector<int64_t> tmpShapeVec = {batchDim*cInDim, dOutDim, dKernelDim, hOutDim, hKernelDim, wOutDim, wKernelDim};
  aclIntArray *shapeArray = executor->AllocIntArray(tmpShapeVec.data(), tmpShapeVec.size());
  OP_CHECK_NULL(shapeArray, return nullptr);
  auto xTensorTmp = l0op::Reshape(xTensor, shapeArray, executor);
  OP_CHECK_NULL(xTensorTmp, return nullptr);
  OP_LOGD("xTensor before permute is: %s", xTensorTmp->ToString().GetString());
  // X: permute(0, 2, 4, 6, 1, 3, 5) from(N*Ci, Do, Dk, Ho, Hk, Wo, Wk) to(N*Ci, Dk, Hk, Wk, Do, Ho, Wo)
  FVector<int64_t> permuteVec = {0, 2, 4, 6, 1, 3, 5};
  aclIntArray *permuteArray = executor->AllocIntArray(permuteVec.data(), permuteVec.size());
  OP_CHECK_NULL(permuteArray, return nullptr);
  xTensorTmp = l0op::Transpose(xTensorTmp, permuteArray, executor);
  OP_CHECK_NULL(xTensorTmp, return nullptr);
  OP_LOGD("xTensor after permute is: %s", xTensorTmp->ToString().GetString());
  // X: reshape from(N*Ci, Dk, Hk, Wk, Do, Ho, Wo) to(N, Ci*Dk*Hk*Wk, Do, Ho, Wo)
  op::Shape tmpShape = op::Shape({
    batchDim, cInDim * dKernelDim * hKernelDim * wKernelDim, dOutDim, hOutDim, wOutDim
  });
  auto xTensorForDw1x1 = ViewWithShape(xTensorTmp, tmpShape, executor);
  OP_CHECK_NULL(xTensorForDw1x1, return nullptr);
  if (xTensorForDw1x1->GetStorageFormat() != xTensor->GetStorageFormat()) {
    xTensorForDw1x1 = l0op::ReFormat(xTensorForDw1x1, xTensor->GetStorageFormat());
  }
  // W: reshape from(Co, Ci, Dk, Hk, Wk) to(Co, Ci*Dk*Hk*Wk, 1, 1, 1)
  tmpShape = op::Shape({
    cOutDim, cInDim * dKernelDim * hKernelDim * wKernelDim, 1, 1, 1
  });
  auto wTensorForDw1x1 = ViewWithShape(wTensor, tmpShape, executor);
  OP_CHECK_NULL(wTensorForDw1x1, return nullptr);
  if (wTensorForDw1x1->GetStorageFormat() != wTensor->GetStorageFormat()) {
    wTensorForDw1x1 = l0op::ReFormat(wTensorForDw1x1, wTensor->GetStorageFormat());
  }
  OP_LOGD("xTensorForDw1x1 is: %s", xTensorForDw1x1->ToString().GetString());
  OP_LOGD("wTensorForDw1x1 is: %s", wTensorForDw1x1->ToString().GetString());
  // 新建一个ConvolutionBackwardInputTensor
  ConvolutionBackwardInputTensor inputTensorForDw1x1 = {gradOutTensor, xTensorForDw1x1, wTensorForDw1x1};
  // 新建一个ConvolutionBackwardParams: stride[1, 1, 1] pad[0, 0, 0]
  int64_t newStrideArray[] = {1, 1, 1};
  int64_t newPaddingArray[] = {0, 0, 0};
  aclIntArray *newStride = executor->AllocIntArray(newStrideArray, CONV3D_ATTR_DIM);
  OP_CHECK_NULL(newStride, return nullptr);
  aclIntArray *newPadding = executor->AllocIntArray(newPaddingArray, CONV3D_ATTR_DIM);
  OP_CHECK_NULL(newPadding, return nullptr);
  ConvolutionBackwardParams paramsForDw1x1 = {params.biasSizes, newStride, newPadding,
                                              params.dilation, params.transposed, params.outputPadding,
                                              params.groups, params.outputMask, params.cubeMathType};
  auto *gradWeight1x1 = l0op::Conv3DBackpropFilter(inputTensorForDw1x1, paramsForDw1x1, executor, useHf32);
  OP_CHECK_NULL(gradWeight1x1, return nullptr);
  OP_LOGD("gradWeight1x1 is: %s", gradWeight1x1->ToString().GetString());
  // gradWeight: reshape from(Co, Ci*Dk*Hk*Wk, 1, 1, 1) to(Co, Ci, Dk, Hk, Wk)
  tmpShape = op::Shape({cOutDim, cInDim, dKernelDim, hKernelDim, wKernelDim});
  auto *gradWeightND = ViewWithShape(gradWeight1x1, tmpShape, executor);
  OP_CHECK_NULL(gradWeightND, return nullptr);
  OP_LOGD("gradWeightND is: %s", gradWeightND->ToString().GetString());
  OP_LOGD("Original ConvBackpropFilter converted to ConvBackpropFilter with kernelD=kernelH=kernelW=1 successfully.");
  return gradWeightND;
}

static aclnnStatus CalculateConv3DBackward(ConvolutionBackwardInputTensor &inputTensor,
                                           ConvolutionBackwardResult &outputTensor, ConvolutionBackwardParams &params,
                                           aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  bool useV2Flag = false;
  bool inputTransDataFlag = false;
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  vector<bool> conv3DBp2MatmulMask {false, false};
  l0op::ConvBackpropParams conv3DBackpropPrarams = {inputTensor.input, inputTensor.weight,
                                                    inputTensor.gradOutput, params.stride,
                                                    params.padding, params.dilation, params.groups};
  auto mmStatus = CalculateConv3DBackwardByMatmulImpl(inputTensor, outputTensor, params, executor, conv3DBp2MatmulMask);
  CHECK_RET(mmStatus == ACLNN_SUCCESS, mmStatus);
  // 如果BpInput和BpFilter都用matmul做完，则提前返回，避免transData耗时.
  if ((*params.outputMask)[0] == conv3DBp2MatmulMask[0] && (*params.outputMask)[1] == conv3DBp2MatmulMask[1]) {
    return ACLNN_SUCCESS;
  }

  if (socVersion != SocVersion::ASCEND910_95) {
    useV2Flag = l0op::IsConv3DBackpropInputV2(conv3DBackpropPrarams);
    inputTransDataFlag = !((*params.outputMask)[1] && l0op::IsConv3DBackpropFilterV2(conv3DBackpropPrarams) && l0op::IsInputTransdataWhiteListCase(conv3DBackpropPrarams)); // dw、v2、白名单同时满足，输入x融合transdata;
    OP_LOGD("Input trans data flag is %d", static_cast<int>(inputTransDataFlag));
  }
  auto promoteType = CalcPromoteType(inputTensor);
  CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor, false, inputTransDataFlag) == ACLNN_SUCCESS,
                ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
            ACLNN_ERR_INNER_NULLPTR);

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());

  bool useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  bool outputTransdataFlag = true;
  if (socVersion == SocVersion::ASCEND910_95) {
    outputTransdataFlag = false;
  }
  if ((*params.outputMask)[0] && !conv3DBp2MatmulMask[0]) {
    OP_LOGD("Enter dx Calculate");
    const aclTensor *gradInputNDC1HWC0 = nullptr;
    if (socVersion == SocVersion::ASCEND910_95) {
      useV2Flag = true;
      gradInputNDC1HWC0 = l0op::Conv3DBackpropInput(inputTensor, params, executor, useHf32);
    } else {
      gradInputNDC1HWC0 = GetGradInputNDC1HWC0(inputTensor, params, executor, useHf32);
    }
    if (useV2Flag && !useHf32 && inputTensor.weight->GetDataType() != DataType::DT_FLOAT) {
      outputTransdataFlag = false; // V2，非HF32，非FP32，不在黑名单
    }
    OP_CHECK(
        gradInputNDC1HWC0 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The calculation with empty tensor failed, Conv3DBackpropInput return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    aclnnStatus status = outputTransdataFlag ? OutputPostProcess(outputTensor.gradInput, gradInputNDC1HWC0, "gradInput",
                                                                 params.groups, executor)
      : OutputPostProcessWithoutTransdata(outputTensor.gradInput, gradInputNDC1HWC0, "gradInput", executor);
    CHECK_RET(status == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  }

  if ((*params.outputMask)[1] && !conv3DBp2MatmulMask[1]) {
    OP_LOGD("Enter dw Calculate");
    if (socVersion == SocVersion::ASCEND910_95) {
      bool transTo1x1DwFlag = IsTransTo1x1Dw(inputTensor, params);
      const aclTensor *gradWeightND = nullptr;
      if (transTo1x1DwFlag) {
        gradWeightND = Conv3DBackpropFilterBy1x1Dw(inputTensor, params, executor, useHf32);
      } else {
        gradWeightND = l0op::Conv3DBackpropFilter(inputTensor, params, executor, useHf32);
      }

      OP_CHECK(gradWeightND != nullptr,
              OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                      "The calculation with empty tensor failed, Conv3dBackpropFilter return nullptr."),
              return ACLNN_ERR_INNER_NULLPTR);
      CHECK_RET(OutputPostProcessWithoutTransdata(outputTensor.gradWeight, gradWeightND, "gradWeight", executor) ==
                ACLNN_SUCCESS,
                ACLNN_ERR_INNER_NULLPTR);
    } else {
          const aclTensor *gradWeightFZ3D = GetGradWeightFZ3D(inputTensor, params, executor, useHf32);
          OP_CHECK(gradWeightFZ3D != nullptr,
                  OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                          "The calculation with empty tensor failed, Conv2dBackpropInput return nullptr."),
                  return ACLNN_ERR_INNER_NULLPTR);
          CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ3D, "gradWeight", params.groups, executor) == ACLNN_SUCCESS,
                    ACLNN_ERR_INNER_NULLPTR);
          }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalcConv3DBackTransposeInputGrad(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params,
                                                    aclOpExecutor *executor,
                                                    bool useHf32) {
  if (!(*params.outputMask)[0]) {
    return ACLNN_SUCCESS;
  }
  OP_LOGD("Enter dx Calculate");
  const aclTensor *gradInputTmp = nullptr;
  aclnnStatus outputPostProcessRet = ACLNN_SUCCESS;
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    if (useHf32) {
      gradInputTmp = l0op::Conv3dv2NCDHWFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, true, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputTmp = l0op::Conv3dv2NCDHWFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, false, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_BF16) {
      gradInputTmp = l0op::Conv3dv2NCDHWBf16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_HIFLOAT8){
      gradInputTmp = l0op::Conv3dv2NCDHWHif8(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputTmp = l0op::Conv3dv2NCDHWFp16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    }
    OP_CHECK(gradInputTmp != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0function result of Conv3dv2NCDHW return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    outputPostProcessRet = OutputPostProcessTransposed(outputTensor.gradInput, gradInputTmp, "gradInput", executor);
  } else {
    if (useHf32) {
      gradInputTmp = l0op::Conv3d6HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, true, executor);
    } else if (inputTensor.weight->GetDataType() == DataType::DT_FLOAT) {
      gradInputTmp = l0op::Conv3d6HdFp32(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, false, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradInputTmp = l0op::Conv3d6HdBf16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    } else {
      gradInputTmp = l0op::Conv3d6HdFp16(inputTensor.gradOutput, inputTensor.weight, nullptr, params.stride,
                                              params.padding, params.dilation, params.groups, executor);
    }
    OP_CHECK(gradInputTmp != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "l0function result of Conv3d6Hd return nullptr."),
        return ACLNN_ERR_INNER_NULLPTR);
    outputPostProcessRet = OutputPostProcess(outputTensor.gradInput, gradInputTmp, "gradInput",
                                              params.groups, executor);
  }
  CHECK_RET(outputPostProcessRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConv3DTransposeBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                    ConvolutionBackwardResult &outputTensor,
                                                    ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  // Index 为 2：进行bias grad运算
  aclnnStatus ret = CalculateBiasGrad(inputTensor, outputTensor, params, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  DataType promoteType;
  bool useHf32 = false;
  if ((*params.outputMask)[1]) {
    promoteType = CalcPromoteType(inputTensor);
    CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(InputPreProcess(inputTensor.input, "input", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  } else if ((*params.outputMask)[0]) {
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        promoteType = CalcPromoteTypeTransposed(inputTensor);
	} else {
        promoteType = CalcPromoteType(inputTensor);
	}
    CHECK_RET(InputPreProcess(inputTensor.gradOutput, "gradOutput", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(InputPreProcess(inputTensor.weight, "weight", params, promoteType, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    useHf32 = ConvBackGoHf32(inputTensor, params.cubeMathType);
  }
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    CHECK_RET(AttrPreProcess(params, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  }

  OP_LOGD("after InputPreProcess with input");
  OP_LOGD("inputTensor.input is: %s", inputTensor.input->ToString().GetString());
  OP_LOGD("inputTensor.weight is: %s", inputTensor.weight->ToString().GetString());
  OP_LOGD("inputTensor.gradOutput is: %s", inputTensor.gradOutput->ToString().GetString());

  if (CalcConv3DBackTransposeInputGrad(inputTensor, outputTensor, params, executor, useHf32) != ACLNN_SUCCESS) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  if ((*params.outputMask)[1]) {
    OP_LOGD("Enter dw Calculate");
    const aclTensor *gradWeightFZ3D = nullptr;
    if (useHf32) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterHf32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input, params.stride,
                                         params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_FLOAT) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp322Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else if (inputTensor.input->GetDataType() == DataType::DT_BF16) {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterBf162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    } else {
      gradWeightFZ3D =
          l0op::Conv3DBackpropFilterFp162Fp32(inputTensor.gradOutput, inputTensor.weight, inputTensor.input,
                                              params.stride, params.padding, params.dilation, params.groups, executor);
    }

    OP_CHECK(gradWeightFZ3D != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The calculation with empty tensor failed, Conv2dBackpropFilter return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
      CHECK_RET(OutputPostProcessWithoutTransdata(outputTensor.gradWeight, gradWeightFZ3D, "gradWeight", executor) ==
                ACLNN_SUCCESS,
                ACLNN_ERR_INNER_NULLPTR);
    } else {
      CHECK_RET(OutputPostProcess(outputTensor.gradWeight, gradWeightFZ3D, "gradWeight", params.groups, executor) ==
                ACLNN_SUCCESS,
                ACLNN_ERR_INNER_NULLPTR);
    }
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CheckCubeMathTypeFor3D(ConvolutionBackwardInputTensor &inputTensor,
  const ConvolutionBackwardParams &params)
{
  DataType promoteDtype = CalcPromoteType(inputTensor);
  int8_t cubeMathType = params.cubeMathType;
  if (cubeMathType == USE_FP16 && promoteDtype == DataType::DT_BF16) {
    OP_LOGW("When promoteDtype is bfloat16, the cubeMathType can not be USE_FP16.");
    return ACLNN_SUCCESS;
  }
  if (cubeMathType == USE_HF32 && promoteDtype == DataType::DT_BF16) {
    OP_LOGW("When promoteDtype is bfloat16, the cubeMathType can not be USE_HF32.");
    return ACLNN_SUCCESS;
  }
  if (cubeMathType == USE_HF32 && promoteDtype == DataType::DT_FLOAT16) {
    OP_LOGW("When promoteDtype is float16, the cubeMathType can not be USE_HF32.");
    return ACLNN_SUCCESS;
  }
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
    if (cubeMathType == ALLOW_FP32_DOWN_PRECISION &&
      (promoteDtype == DataType::DT_FLOAT8_E4M3FN || promoteDtype == DataType::DT_HIFLOAT8)) {
      OP_LOGW("When promoteDtype(%s) is hifloat8/float8_e4m3fn, the cubeMathType can not be ALLOW_FP32_DOWN_PRECISION.",
        op::ToString(promoteDtype).GetString());
      return ACLNN_SUCCESS;
    }
    if (cubeMathType == USE_FP16 &&
      (promoteDtype == DataType::DT_FLOAT8_E4M3FN || promoteDtype == DataType::DT_HIFLOAT8)) {
      OP_LOGW("When promoteDtype(%s) is hifloat8/float8_e4m3fn, the cubeMathType can not be USE_FP16.",
        op::ToString(promoteDtype).GetString());
      return ACLNN_SUCCESS;
    }
    if (cubeMathType == USE_HF32 &&
      (promoteDtype == DataType::DT_FLOAT8_E4M3FN || promoteDtype == DataType::DT_HIFLOAT8)) {
      OP_LOGW("When promoteDtype(%s) is hifloat8/float8_e4m3fn, the cubeMathType can not be USE_HF32.",
        op::ToString(promoteDtype).GetString());
      return ACLNN_SUCCESS;
    }
  }
  return ACLNN_SUCCESS;
}

static bool isConv2dTo3d(const ConvolutionBackwardInputTensor &inputTensor,
                         const ConvolutionBackwardParams &params) {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
  if (inputDim != CONV2DINPUTDIM) {
    return false;
  }

  // 910_95: 2D->3D
  if (socVersion == SocVersion::ASCEND910_95) {
    return true;
  }

  if (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93) {
    // 满足白名单时2D->3D
    l0op::ConvBackpropParams conv2DBackpropParams = {inputTensor.input, inputTensor.weight, inputTensor.gradOutput,
      params.stride, params.padding, params.dilation, params.groups};
    bool gradInputWhiteListCase = l0op::IsConv2DBackpropInputTo3DCase(conv2DBackpropParams);
    bool gradWeightWhiteListCase = l0op::IsConv2DBpFilterTo3Dcase(conv2DBackpropParams);
    return gradInputWhiteListCase || gradWeightWhiteListCase;
  }

  return false;
}

static aclnnStatus CalculateConv3DBp(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardResult &outputTensor,
                                     ConvolutionBackwardParams &params,
                                     aclOpExecutor *executor)
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  OP_CHECK(socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
           socVersion == SocVersion::ASCEND910_95,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not implemented for %s", op::ToString(socVersion).GetString()),
           return ACLNN_ERR_PARAM_INVALID);
  auto ret = CheckCubeMathTypeFor3D(inputTensor, params);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  if (socVersion != SocVersion::ASCEND910_95) {
    CHECK_RET(CheckSupportedForConv3dBackpropFilter(inputTensor, outputTensor, params), ACLNN_ERR_PARAM_INVALID);
  }
  if (!params.transposed) {
    OP_LOGD("Entering CalculateConv3DBackward");
    ret = CalculateConv3DBackward(inputTensor, outputTensor, params, executor);
  } else {
    OP_LOGD("Entering CalculateConv3DTransposeBackward");
    ret = CalculateConv3DTransposeBackward(inputTensor, outputTensor, params, executor);
  }

  return ret;
}

static aclnnStatus CalculateConv2DBp(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardResult &outputTensor,
                                     ConvolutionBackwardParams &params,
                                     aclOpExecutor *executor)
{
  // 只支持NCDHW格式
  bool isConv2DTo3D = isConv2dTo3d(inputTensor, params);
  if(isConv2DTo3D) {
    OP_LOGD("ConvolutionBackward change 2D to 3D");
    inputTensor.input = View4Das5D(inputTensor.input, executor);
    OP_CHECK(inputTensor.input != nullptr, OP_LOGD("input squeeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    inputTensor.gradOutput = View4Das5D(inputTensor.gradOutput, executor);
    OP_CHECK(inputTensor.gradOutput != nullptr, OP_LOGD("gradOutput squeeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    inputTensor.weight = View4Das5D(inputTensor.weight, executor);
    OP_CHECK(inputTensor.weight != nullptr, OP_LOGD("weight squeeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    if ((*params.outputMask)[0]) { // 计算dx时
      outputTensor.gradInput = View4Das5D(outputTensor.gradInput, executor);
      OP_CHECK(outputTensor.gradInput != nullptr, OP_LOGD("gradInput squeeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    }
    if ((*params.outputMask)[1]) { // 计算dw时
      outputTensor.gradWeight = View4Das5D(outputTensor.gradWeight, executor);
      OP_CHECK(outputTensor.gradWeight != nullptr, OP_LOGD("gradWeight squeeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    }
    FVector<int64_t> newStrides = {1, (*params.stride)[0], (*params.stride)[1]};
    FVector<int64_t> newDilation = {1, (*params.dilation)[0], (*params.dilation)[1]};
    FVector<int64_t> newPad {0, 0, (*params.padding)[0], (*params.padding)[0], (*params.padding)[1], (*params.padding)[1]};
    if (params.padding->Size() == 4) { // padding为4维, [up, down, left, right]
      newPad = {0, 0, (*params.padding)[0], (*params.padding)[1], (*params.padding)[2], (*params.padding)[3]};
    }
    params.stride = executor->AllocIntArray(newStrides.data(), newStrides.size());
    OP_CHECK(params.stride != nullptr, OP_LOGD("newStrides alloc failed."), return ACLNN_ERR_INNER_NULLPTR);
    params.dilation = executor->AllocIntArray(newDilation.data(), newDilation.size()); // conv3D Dilation dim = 5;
    OP_CHECK(params.dilation != nullptr, OP_LOGD("newDilation alloc failed."), return ACLNN_ERR_INNER_NULLPTR);
    params.padding = executor->AllocIntArray(newPad.data(), newPad.size()); // conv3D Pad dim = 6;
    OP_CHECK(params.padding!= nullptr, OP_LOGD("newPad alloc failed."), return ACLNN_ERR_INNER_NULLPTR);

    auto ret = CalculateConv3DBp(inputTensor, outputTensor, params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    if ((*params.outputMask)[0]) {
      outputTensor.gradInput = View5Das4D(outputTensor.gradInput, executor);
      OP_CHECK(outputTensor.gradInput != nullptr, OP_LOGD("gradInput unsqueeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    }
    if ((*params.outputMask)[1]) {
      outputTensor.gradWeight = View5Das4D(outputTensor.gradWeight, executor);
      OP_CHECK(outputTensor.gradWeight != nullptr, OP_LOGD("gradWeight unsqueeze failed."), return ACLNN_ERR_INNER_NULLPTR);
    }

    return ret;
  }

  if (!params.transposed) {
    OP_LOGD("Entering CalculateConv2DBackward");
    return CalculateConv2DBackward(inputTensor, outputTensor, params, executor);
  }

  OP_LOGD("Entering CalculateConv2DTransposeBackward");
  return CalculateConv2DTransposeBackward(inputTensor, outputTensor, params, executor);
}
}

static aclnnStatus CalculateConvolutionBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                ConvolutionBackwardOutput &outputTensor,
                                                ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  ConvolutionBackwardResult resultTensor = {outputTensor.gradInput, outputTensor.gradWeight, outputTensor.gradBias};
  auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();

  // inputDim 为 3 时，进行1D卷积反向
  if (inputDim == CONV1DINPUTDIM) {
    if (socVersion == SocVersion::ASCEND910_95) {
      auto ret = CheckCubeMathTypeFor3D(inputTensor, params);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv1DBackward");
      auto ret = CalculateConv1DBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    } else {
      OP_LOGD("Entering CalculateConv1DTransposeBackward");
      auto ret = CalculateConv1DTransposeBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    }
    // inputDim 为 4 时，非ASCEND910_95平台进行2D卷积反向
  } else if (inputDim == CONV2DINPUTDIM) {
    auto ret = CalculateConv2DBp(inputTensor, resultTensor, params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  } else if (inputDim == CONV3DINPUTDIM) {
    auto ret = CalculateConv3DBp(inputTensor, resultTensor, params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "ConvolutionBackward only support input with dimensions 3, 4, or 5, Actually is %ld.", inputDim);
    return ACLNN_ERR_PARAM_INVALID;
  }
  auto ret = OutputViewProcess(resultTensor, outputTensor, params.outputMask, executor);
  CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  OP_LOGD("After CalculateConvolutionBackward");
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionTbcBackwardWithEmpty(aclTensor *&gradInput, aclTensor *&gradWeight,
                                                            aclTensor *&gradBias, const int64_t pad,
                                                            aclOpExecutor *executor) {
  if (gradInput->Size() != 0) {
    auto inputValueScalar = executor->AllocScalar(0);
    auto inputValueTensor = executor->ConvertToTensor(inputValueScalar, gradInput->GetDataType());
    auto inputFillShape = op::ToShapeVector(gradInput->GetViewShape());
    aclIntArray *inputShapeArray = executor->AllocIntArray(inputFillShape.data(), inputFillShape.size());
    CHECK_RET(inputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *inputDims = executor->ConvertToTensor(inputFillShape.data(),
                                                           inputFillShape.size(), op::DataType::DT_INT64);
    auto gradInputZeros = l0op::Fill(inputDims, inputValueTensor, inputShapeArray, executor);
    CHECK_RET(gradInputZeros != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradInputResult = l0op::ViewCopy(gradInputZeros, gradInput, executor);
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (gradWeight->Size() != 0) {
    auto weightValueScalar = executor->AllocScalar(0);
    auto weightValueTensor = executor->ConvertToTensor(weightValueScalar, gradWeight->GetDataType());
    auto weightFillShape = op::ToShapeVector(gradWeight->GetViewShape());
    aclIntArray *weightShapeArray = executor->AllocIntArray(weightFillShape.data(), weightFillShape.size());
    CHECK_RET(weightShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *weightDims = executor->ConvertToTensor(weightFillShape.data(),
                                                           weightFillShape.size(), op::DataType::DT_INT64);
    CHECK_RET(weightDims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradWeightZeros = l0op::Fill(weightDims, weightValueTensor, weightShapeArray, executor);
    CHECK_RET(gradWeightZeros != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradWeightResult = l0op::ViewCopy(gradWeightZeros, gradWeight, executor);
    CHECK_RET(gradWeightResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  if (gradBias->Size() != 0) {
    auto t = gradInput->GetViewShape().GetDim(0) + 2 * pad + 1 - gradWeight->GetViewShape().GetDim(0);
    float value = 1.0f * gradInput->GetViewShape().GetDim(1) * t;
    auto valueScalar = executor->AllocScalar(value);
    auto valueTensor = executor->ConvertToTensor(valueScalar, gradBias->GetDataType());
    auto fillShape = op::ToShapeVector(gradBias->GetOriginalShape());
    aclIntArray *shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
    CHECK_RET(shapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
    CHECK_RET(dims != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradBiasOnes = l0op::Fill(dims, valueTensor, shapeArray, executor);
    CHECK_RET(gradBiasOnes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradBiasResult = l0op::ViewCopy(gradBiasOnes, gradBias, executor);
    CHECK_RET(gradBiasResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  return ACLNN_SUCCESS;
}

static aclnnStatus CalculateConvolutionTbcBackward(ConvolutionBackwardInputTensor &inputTensor,
                                                   ConvolutionBackwardOutput &finalTensor,
                                                   ConvolutionBackwardParams &params, aclOpExecutor *executor) {
  aclTensor *gradInputNcl = executor->AllocTensor(inputTensor.input->GetViewShape(), inputTensor.input->GetDataType());
  aclTensor *gradWeightNcl = executor->AllocTensor(inputTensor.weight->GetViewShape(),
                                                   inputTensor.weight->GetDataType());
  aclTensor *gradBiasNcl = executor->AllocTensor(finalTensor.gradBias->GetViewShape(),
                                                 finalTensor.gradBias->GetDataType());
  ConvolutionBackwardResult resultTensor = {gradInputNcl, gradWeightNcl, gradBiasNcl};
  auto inputDim = inputTensor.input->GetViewShape().GetDimNum();
  // inputDim 为 3 时，进行1D卷积反向
  if (inputDim == 3) {
    if (!params.transposed) {
      OP_LOGD("Entering CalculateConv1DBackward");
      auto ret = CalculateConv1DBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    } else {
      OP_LOGD("Entering CalculateConv1DTransposeBackward");
      auto ret = CalculateConv1DTransposeBackward(inputTensor, resultTensor, params, executor);
      CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    }
  } else {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "ConvolutionTbcBackward only support input with dimensions 3, Actually is %ld.", inputDim);
    return ACLNN_ERR_INNER_NULLPTR;
  }
  // recover NCL to TBC
  vector<int64_t> gradInputDims = {2, 0, 1};
  vector<int64_t> weightDims = {2, 1, 0};
  auto gradInputTbc = Permute(resultTensor.gradInput, gradInputDims, executor);
  CHECK_RET(gradInputTbc != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradWeightTbc = Permute(resultTensor.gradWeight, weightDims, executor);
  CHECK_RET(gradWeightTbc != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradBiasTbc = resultTensor.gradBias;
  // transdata: change result format --> out format
  if (gradInputTbc->GetStorageFormat() != finalTensor.gradInput->GetStorageFormat()) {
    gradInputTbc = l0op::ReFormat(gradInputTbc, finalTensor.gradInput->GetStorageFormat());
  }
  if (gradWeightTbc->GetStorageFormat() != finalTensor.gradWeight->GetStorageFormat()) {
    gradWeightTbc = l0op::ReFormat(gradWeightTbc, finalTensor.gradWeight->GetStorageFormat());
  }

  // viewcopy
  auto gradInputResult = l0op::ViewCopy(gradInputTbc, finalTensor.gradInput, executor);
  CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradWeightResult = l0op::ViewCopy(gradWeightTbc, finalTensor.gradWeight, executor);
  CHECK_RET(gradWeightResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto gradBiasResult = l0op::ViewCopy(gradBiasTbc, finalTensor.gradBias, executor);
  CHECK_RET(gradBiasResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed,
    const aclIntArray *outputPadding, int groups, const aclBoolArray *outputMask, int8_t cubeMathType,
    aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnConvolutionBackward,
                   DFX_IN(gradOutput, input, weight, biasSizes, stride, padding, dilation, transposed, outputPadding,
                          groups, outputMask, cubeMathType),
                   DFX_OUT(gradInput, gradWeight, gradBias));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ConvolutionBackwardInputTensor inputTensor = {gradOutput, input, weight};
    ConvolutionBackwardOutput outputTensor = {gradInput, gradWeight, gradBias};
    ConvolutionBackwardParams params = {biasSizes,     stride, padding,    dilation,    transposed,
                                        outputPadding, groups, outputMask, cubeMathType};

    // 固定写法，参数检查
    const SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    Ops::NN::Conv::ConvolutionBackwardChecker convolutionBackwardChecker(inputTensor, outputTensor, params, socVersion);
    auto ret = convolutionBackwardChecker.CheckParams();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查conv3ddw确定性计算
    if ((*outputMask)[1] && (input->GetViewShape().GetDimNum() == CONV3DINPUTDIM ||
      (input->GetViewShape().GetDimNum() == CONV2DINPUTDIM && socVersion == SocVersion::ASCEND910_95))) {
      int64_t deterministicValue = 0;
      rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
      if (retRts != RT_ERROR_NONE) {
        deterministicValue = 0;
      }
      if (socVersion != SocVersion::ASCEND910_95 && socVersion != SocVersion::ASCEND910B && socVersion != SocVersion::ASCEND910_93) {
        CHECK_RET(CheckDeterministic(deterministicValue, groups), ACLNN_ERR_PARAM_INVALID);
      }
    }

    if (input->Size() == 0 || weight->Size() == 0 || gradOutput->Size() == 0) {
        OP_LOGD("Entering CalculateConvolutionBackwardWithEmpty");
        CHECK_RET(convolutionBackwardChecker.CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);
        ret = CalculateConvolutionBackwardWithEmpty(inputTensor, outputTensor, params, uniqueExecutor.get());
    } else {
        OP_LOGD("Entering CalculateConvolutionBackward 0419");
        ret = CalculateConvolutionBackward(inputTensor, outputTensor, params, uniqueExecutor.get());
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

namespace {
struct ReshapeTargetConf {
    std::vector<int64_t> dims;
    op::Format targetFormat;
    const string tensorName;
};

static const aclTensor *ReshapeTensor(const aclTensor *input, aclOpExecutor *executor, const ReshapeTargetConf &conf)
{
    auto contiguousInput = l0op::Contiguous(input, executor);
    CHECK_RET(contiguousInput != nullptr, nullptr);
    aclIntArray *dimsArray = executor->AllocIntArray(conf.dims.data(), conf.dims.size());
    CHECK_RET(dimsArray != nullptr, nullptr);
    auto reshapeInput = l0op::Reshape(contiguousInput, dimsArray, executor);
    OP_CHECK(reshapeInput != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The Reshape with %s failed.", conf.tensorName.c_str()),
             return nullptr);
    auto finalInput = l0op::ReFormat(reshapeInput, conf.targetFormat);
    OP_CHECK(finalInput != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The ReFormat with %s failed.", conf.tensorName.c_str()),
             return nullptr);
    return finalInput;
}

static const aclTensor *ReshapeTBCTo3DFormat(const aclTensor *input, aclOpExecutor *executor, const string &tensorName)
{
    ReshapeTargetConf conf = {
        .dims = {1, 1, input->GetViewShape().GetDim(0), input->GetViewShape().GetDim(1), -1}, // [1, 1, T, B, C]
        .targetFormat = op::Format::FORMAT_NDHWC,
        .tensorName = tensorName
    };
    if (tensorName == "weight" || tensorName == "gradWeight") {
        conf.dims = {1, input->GetViewShape().GetDim(0), 1, input->GetViewShape().GetDim(1), -1}; // [1, K, 1, Cin, Cout]
        conf.targetFormat = op::Format::FORMAT_DHWCN;
    }
    return ReshapeTensor(input, executor, conf);
}

static const aclTensor *Reshape3DFormatToTBC(const aclTensor *input, const op::Format &format, aclOpExecutor *executor,
                                          const string &tensorName)
{
    ReshapeTargetConf conf = {
        .dims = {input->GetViewShape().GetDim(2), input->GetViewShape().GetDim(3), -1}, // [T, B, C]
        .targetFormat = format,
        .tensorName = tensorName
    };
    if (tensorName == "gradWeight") {
        conf.dims = {input->GetViewShape().GetDim(1), input->GetViewShape().GetDim(3), -1}; // [K, Cin, Cout]
    }
    return ReshapeTensor(input, executor, conf);
}

static aclnnStatus CalculateConvolutionTbcBackwardBy3D(ConvolutionBackwardInputTensor &inputTensor,
                                                   ConvolutionBackwardOutput &finalTensor,
                                                   ConvolutionBackwardParams &params, aclOpExecutor *executor)
{
    aclTensor *gradInput = executor->AllocTensor(inputTensor.input->GetViewShape(), inputTensor.input->GetDataType());
    aclTensor *gradWeight = executor->AllocTensor(inputTensor.weight->GetViewShape(), inputTensor.weight->GetDataType());
    aclTensor *gradBias = executor->AllocTensor(finalTensor.gradBias->GetViewShape(), finalTensor.gradBias->GetDataType());

    ConvolutionBackwardResult resultTensor = {gradInput, gradWeight, gradBias};

    // T 当作 H 轴
    FVector<int64_t> padding = {0, (*params.padding)[0], 0};
    FVector<int64_t> stride = {1, 1, 1};
    FVector<int64_t> dilation = {1, 1, 1};
    params.padding = executor->AllocIntArray(padding.data(), padding.size());
    CHECK_RET(params.padding != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.stride = executor->AllocIntArray(stride.data(), stride.size());
    CHECK_RET(params.stride != nullptr, ACLNN_ERR_INNER_NULLPTR);
    params.dilation = executor->AllocIntArray(dilation.data(), dilation.size());
    CHECK_RET(params.dilation != nullptr, ACLNN_ERR_INNER_NULLPTR);

    inputTensor.input = ReshapeTBCTo3DFormat(inputTensor.input, executor, "input");
    CHECK_RET(inputTensor.input != nullptr, ACLNN_ERR_INNER_NULLPTR);
    inputTensor.gradOutput = ReshapeTBCTo3DFormat(inputTensor.gradOutput, executor, "gradOutput");
    CHECK_RET(inputTensor.gradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    inputTensor.weight = ReshapeTBCTo3DFormat(inputTensor.weight, executor, "weight");
    CHECK_RET(inputTensor.weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resultTensor.gradInput = ReshapeTBCTo3DFormat(resultTensor.gradInput, executor, "gradInput");
    CHECK_RET(resultTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resultTensor.gradWeight = ReshapeTBCTo3DFormat(resultTensor.gradWeight, executor, "gradWeight");
    CHECK_RET(resultTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    resultTensor.gradBias = gradBias;

    auto ret = CalculateConv3DBackward(inputTensor, resultTensor, params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if ((*params.outputMask)[0]) {
        resultTensor.gradInput = Reshape3DFormatToTBC(resultTensor.gradInput, finalTensor.gradInput->GetStorageFormat(),
                                                   executor, "gradInput");
        CHECK_RET(resultTensor.gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if ((*params.outputMask)[1]) {
        resultTensor.gradWeight = Reshape3DFormatToTBC(resultTensor.gradWeight, finalTensor.gradWeight->GetStorageFormat(),
                                                   executor, "gradWeight");
        CHECK_RET(resultTensor.gradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto gradInputResult = l0op::ViewCopy(resultTensor.gradInput, finalTensor.gradInput, executor);
    OP_CHECK(gradInputResult != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The tbc output viewprocess failed, gradInput with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto gradWeightResult = l0op::ViewCopy(resultTensor.gradWeight, finalTensor.gradWeight, executor);
    OP_CHECK(gradWeightResult != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The tbc output viewprocess failed, gradWeight with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    auto gradBiasResult = l0op::ViewCopy(resultTensor.gradBias, finalTensor.gradBias, executor);
    OP_CHECK(gradBiasResult != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The tbc output viewprocess failed, gradBias with ViewCopy return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor *self, const aclTensor *input, const aclTensor *weight,
                                                 const aclTensor *bias, int64_t pad, int8_t cubeMathType,
                                                 aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
                                                 uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnConvTbcBackward, DFX_IN(self, input, weight, bias, pad, cubeMathType),
                   DFX_OUT(gradInput, gradWeight, gradBias));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    Ops::NN::Conv::ConvTbcBackwardInput inputTensor = {self, input, weight, bias};
    Ops::NN::Conv::ConvTbcBackwardOutput outputTensor = {gradInput, gradWeight, gradBias};
    Ops::NN::Conv::ConvTbcBackwardParams tbcparams = {pad, cubeMathType};
    // 固定写法，参数检查
    const SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    Ops::NN::Conv::ConvTbcBackwardChecker convTbcBackwardChecker(inputTensor, outputTensor, tbcparams, socVersion);
    auto ret = convTbcBackwardChecker.CheckTbcParams();
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 设置param
    FVector<int64_t> newStride = {1};
    FVector<int64_t> newPadding = {pad};
    FVector<int64_t> newDilation = {1};
    FVector<int64_t> newOutputPadding = {0};
    FVector<bool> newOutputMask = {1, 1, 1};
    FVector<int64_t> newBias = {bias->Size()};
    auto *stride = uniqueExecutor.get()->AllocIntArray(newStride.data(), 1);
    auto *padding = uniqueExecutor.get()->AllocIntArray(newPadding.data(), 1);
    auto *dilation = uniqueExecutor.get()->AllocIntArray(newDilation.data(), 1);
    auto *outputPadding = uniqueExecutor.get()->AllocIntArray(newOutputPadding.data(), 1);
    auto *outputMask = uniqueExecutor.get()->AllocBoolArray(newOutputMask.data(), 3);
    auto biasSizes = uniqueExecutor.get()->AllocIntArray(newBias.data(), 1);
    ConvolutionBackwardParams params = {biasSizes,     stride, padding,    dilation, 0,
                                        outputPadding, 1, outputMask, cubeMathType}; // transposed = 0， groups = 1
    ConvolutionBackwardOutput finalTensor = {gradInput, gradWeight, gradBias};

    if (input->Size() == 0 || weight->Size() == 0 || bias->Size() == 0) {
        OP_LOGD("Entering CalculateConvolutionTbcBackwardWithEmpty");
        ret = CalculateConvolutionTbcBackwardWithEmpty(gradInput, gradWeight, gradBias, pad, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
         ConvolutionBackwardInputTensor inputTensorTbc = {self, input, weight};
         OP_LOGD("Entering CalculateConvolutionTbcBackwardBy3D");
         ret = CalculateConvolutionTbcBackwardBy3D(inputTensorTbc, finalTensor, params, uniqueExecutor.get());
         CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        // transpose TBC to NCL
        vector<int64_t> dims = {1, 2, 0};
        vector<int64_t> weightDims = {2, 1, 0};
        auto selfPermuted = Permute(self, dims, uniqueExecutor.get());
        CHECK_RET(selfPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto inputPermuted = Permute(input, dims, uniqueExecutor.get());
        CHECK_RET(inputPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto weightPermuted = Permute(weight, weightDims, uniqueExecutor.get());
        CHECK_RET(weightPermuted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        ConvolutionBackwardInputTensor inputTensorNcl = {selfPermuted, inputPermuted, weightPermuted};
        OP_LOGD("Entering CalculateConvolutionTbcBackward");
        ret = CalculateConvolutionTbcBackward(inputTensorNcl, finalTensor, params, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvolutionBackward);
    OP_LOGD("Entering aclnnConvolutionBackward");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnConvTbcBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConvTbcBackward);
    OP_LOGD("Entering aclnnConvTbcBackward");
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
