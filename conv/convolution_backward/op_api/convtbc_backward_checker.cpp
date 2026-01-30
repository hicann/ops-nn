/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "convtbc_backward_checker.h"

using namespace op;

namespace Ops {
namespace NN {
namespace Conv {

inline bool ConvTbcBackwardChecker::CheckTbcNotNull() {
    OP_CHECK_NULL(inputTensor_.self, return false);
    OP_CHECK_NULL(inputTensor_.input, return false);
    OP_CHECK_NULL(inputTensor_.weight, return false);
    OP_CHECK_NULL(inputTensor_.bias, return false);
    OP_CHECK_NULL(outputTensor_.gradInput, return false);
    OP_CHECK_NULL(outputTensor_.gradWeight, return false);
    OP_CHECK_NULL(outputTensor_.gradBias, return false);
    return true;
}

bool ConvTbcBackwardChecker::CheckTbcDtypeValid(const aclTensor *inputTensor) const {
  // 检查输入aclTensor的数据类型是否在ConvolutionBackward支持列表内
  if (npuArch_ == NpuArch::DAV_3510) {
    auto dtypeSupportList = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  } else {
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  }
  return true;
}

bool ConvTbcBackwardChecker::CheckDtypeValidBf16Allowed(const aclTensor *inputTensor) const {
  // 检查输入aclTensor的数据类型是否在ConvolutionBackward支持列表内
  auto dtypeSupportList = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
  OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor, dtypeSupportList, return false);
  return true;
}

bool ConvTbcBackwardChecker::CheckTbcFormat(const aclTensor *inputTensor, const string &tensorName) const {
    OP_CHECK(inputTensor->GetStorageFormat() == op::Format::FORMAT_ND ||
                 inputTensor->GetStorageFormat() == op::Format::FORMAT_NCL,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s format only support ND or NCL, but got %s.", tensorName.c_str(),
                     op::ToString(inputTensor->GetStorageFormat()).GetString()),
             return false);
    return true;
}

bool ConvTbcBackwardChecker::CheckTbcBiasFormat(const aclTensor *inputTensor, const string &tensorName) const {
    OP_CHECK(inputTensor->GetStorageFormat() == op::Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s format only support ND, but got %s.", tensorName.c_str(),
                     op::ToString(inputTensor->GetStorageFormat()).GetString()),
             return false);
    return true;
}

bool ConvTbcBackwardChecker::CheckTbcShape() {
  auto validDim = [](const aclTensor *tensor, int64_t dims, const char* paramName) -> bool {
    int64_t curDims = tensor->GetViewShape().GetDimNum();
    OP_CHECK(curDims == dims,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the dimension of %s should be %ld, but get %ld.",
                paramName, dims, curDims),
        return false);
    return true;
  };
  constexpr int64_t tbcDims = 3;
  // self，weight，input, gradInput, gradWeight只能是3维，bias, gradBias只能是1维
  bool res = validDim(inputTensor_.self, tbcDims, "self") && validDim(inputTensor_.input, tbcDims, "input")
             && validDim(inputTensor_.weight, tbcDims, "weight") && validDim(inputTensor_.bias, 1, "bias")
             && validDim(outputTensor_.gradBias, 1, "gradBias") && validDim(outputTensor_.gradInput, tbcDims, "gradInput")
             && validDim(outputTensor_.gradWeight, tbcDims, "gradWeight");
  if (!res) {
    return false;
  }

  // input(TBC1) weigth(LC1C0) bias(C0):
  OP_CHECK(inputTensor_.input->GetViewShape().GetDim(2) == inputTensor_.weight->GetViewShape().GetDim(1),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dim 2 (Input Channels) is not == dim 1 in the weight tensor."),
           return false);
  OP_CHECK(inputTensor_.bias->GetViewShape().GetDim(0) == inputTensor_.weight->GetViewShape().GetDim(2),
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias should be [%ld], but get [%ld]",
                   inputTensor_.weight->GetViewShape().GetDim(2), inputTensor_.bias->GetViewShape().GetDim(0)),
           return false);
  // pad >= 0
  OP_CHECK(params_.pad >= 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The pad must be greater than or equal to 0, but get %ld.", params_.pad),
           return false);
  // self 与input, weight shape 必须满足约束
  // 约束1：self shape必须与conv_tbc计算出的output match： self(T+2*pad+1-L,B,C0)
  auto t = inputTensor_.input->GetViewShape().GetDim(0) + 2 * params_.pad + 1 - inputTensor_.weight->GetViewShape().GetDim(0);
  auto b = inputTensor_.input->GetViewShape().GetDim(1);
  auto c0 = inputTensor_.weight->GetViewShape().GetDim(2);
  OP_CHECK(t >= 0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
           "Try to create tensor with negative dimension %ld:[%ld, %ld, %ld]",
           t, t, b, c0),
           return false);
  OP_CHECK(inputTensor_.self->GetViewShape().GetDim(0) == t && inputTensor_.self->GetViewShape().GetDim(1) == b &&
           inputTensor_.self->GetViewShape().GetDim(2) == c0,
           OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                   "Mismatch in shape: grad_output has a shape of %s but output has a shape of [%ld, %ld, %ld],"
                   "which output shape is deduced from the input and the weight",
                   op::ToString(inputTensor_.self->GetViewShape()).GetString(), t, b, c0),
           return false);

  // input和gradInput的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(inputTensor_.input, outputTensor_.gradInput, return false);
  // weight和gradWeight的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(inputTensor_.weight, outputTensor_.gradWeight, return false);
  // bias和gradBias的shape必须一致
  OP_CHECK_SHAPE_NOT_EQUAL(inputTensor_.bias, outputTensor_.gradBias, return false);
  return true;
}

bool ConvTbcBackwardChecker::CheckTbcCubeMathType() {
  // 计算promote dtype
  auto gradOutputDtype = inputTensor_.self->GetDataType();
  auto inputDtype = inputTensor_.input->GetDataType();
  auto weightDtype = inputTensor_.weight->GetDataType();
  auto promoteType1 = op::PromoteType(gradOutputDtype, inputDtype);
  auto promoteTypeFinal = op::PromoteType(promoteType1, weightDtype);
  return CheckCubeMathType(promoteTypeFinal, params_.cubeMathType);
}

aclnnStatus ConvTbcBackwardChecker::CheckTbcParams() {
    // 检查ConvolutionTbcBackward的输入aclTensor是否符合规范
    // 1. 检查输入aclTensor是否为空指针
    CHECK_RET(CheckTbcNotNull(), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入aclTensor的Dtype是否在API支持的数据类型范围之内
    CHECK_RET(CheckTbcDtypeValid(inputTensor_.self), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcDtypeValid(inputTensor_.input), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcDtypeValid(inputTensor_.weight), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValidBf16Allowed(inputTensor_.bias), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcDtypeValid(outputTensor_.gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcDtypeValid(outputTensor_.gradWeight), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcDtypeValid(outputTensor_.gradBias), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入aclTensor的Format是否在API支持的数据类型范围之内
    CHECK_RET(CheckTbcFormat(inputTensor_.self, "Self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(inputTensor_.input, "Input"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(inputTensor_.weight, "Weight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcBiasFormat(inputTensor_.bias, "bias"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(outputTensor_.gradInput, "gradInput"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcFormat(outputTensor_.gradWeight, "gradWeight"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTbcBiasFormat(outputTensor_.gradBias, "gradBias"), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self不能为空
    OP_CHECK(inputTensor_.self->Size() != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The self can not be empty tensor."),
             return ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入aclTensor的shape是否符合约束
    CHECK_RET(CheckTbcShape(), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查cubeMathType
    CHECK_RET(CheckTbcCubeMathType(), ACLNN_ERR_PARAM_INVALID);

    if (npuArch_ == NpuArch::DAV_3510) {
       // 检查pad是否在[0, 255]之间
      OP_CHECK(params_.pad >= 0 && params_.pad <= 255, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pad value [%ld] is invalid, support range [0, 255].", params_.pad),
               return ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

} // namespace Conv
} // namespace NN
} // namespace Ops