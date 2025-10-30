/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_avgpool2d_backward.cpp
 * \brief
 */

#include "aclnn_avgpool2d_backward.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/fp16_t.h"
#include "opdev/framework_op.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/div.h"
#include "level0/mul.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "avgpool3d_backward.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "op_api/op_api_def.h"
#include "avgpool2d_conv2d_backprop_input_util.h"

using namespace avgpool2d_conv2d_input_util;
using namespace op;
namespace op {
static aclnnStatus CheckArrayDataAvgPoolBackWard(
    const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding)
{
    for (uint64_t i = 0; i < kernelSize->Size(); i++) {
        auto size = (*kernelSize)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize [%lu] data [%li] is less than or equal to 0", i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    for (uint64_t i = 0; i < stride->Size(); i++) {
        auto size = (*stride)[i];
        if (size <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride [%lu] data [%li] is less than or equal to 0", i, size);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (kernelSize->Size() != padding->Size()) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    for (uint64_t i = 0; i < kernelSize->Size(); i++) {
        auto halfKsize = (*kernelSize)[i] / 2;
        auto padSize = (*padding)[i];
        if (halfKsize < padSize) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "padding [%lu] data [%li] should less than kernelSize / 2 [%ld]", i, padSize,
                halfKsize);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (padSize < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding [%lu] data [%li] is less than 0", i, padSize);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}
static bool CheckPaddingValidAvgPool2D(const aclIntArray *kernelSize, const aclIntArray *padding) {
  const int64_t kKernelSizeHIdx = 0;
  const int64_t kKernelSizeWIdx = 1;
  const int64_t kPaddingUpIdx = 0;
  const int64_t kPaddingLeftIdx = 1;
  const int64_t kernelH = (*kernelSize)[kKernelSizeHIdx];
  const int64_t kernelPaddingSize = 1;
  const int64_t MULTIPLIER = 2;
  // 1表示kernelSize长度为1
  const int64_t kernelW = kernelSize->Size() == kernelPaddingSize ? (*kernelSize)[kKernelSizeHIdx] : (*kernelSize)[kKernelSizeWIdx];
  OP_CHECK(padding->Size() != 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding is empty"), return false);
  const int64_t paddingH = (*padding)[kPaddingUpIdx];
  const int64_t paddingW = padding->Size() == kernelPaddingSize ? (*padding)[kPaddingUpIdx] : (*padding)[kPaddingLeftIdx];
  // MULTIPLIER 表示paddingH不能大于kernelH的一半，下同
  if (kernelH < MULTIPLIER * paddingH) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "value of paddingH should be at most half of kernelH. Actual: paddingH is [%ld],kernelH is [%ld].",
            paddingH, kernelH);
    return false;
  }
  if (kernelW < MULTIPLIER * paddingW) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "value of paddingW should be at most half of kernelW. Actual: paddingW is [%ld],"
            "kernelW is [%ld].",
            paddingW, kernelW);
    return false;
  }
  return true;
}

} // namespace op
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_WITH_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

const int64_t kDivisoroverrideMaxValue = 255;
const int64_t kDivisoroverrideMinValue = -255L;
const int kNDimNCHWIdx = 0;
const int kCDimNCHWIdx = 1;
const int kHDimNCHWIdx = 2;
const int kWDimNCHWIdx = 3;
const int kCDimNCLIdx = 0;
const int kHDimNCLIdx = 1;
const int kWDimNCLIdx = 2;
const int kKernelSizeHIdx = 0;
const int kKernelSizeWIdx = 1;
const int kStrideHIdx = 0;
const int kStrideWIdx = 1;
const int kPaddingUpIdx = 0;
const int kPaddingLeftIdx = 1;
const int kRealPaddingUpIdx = 0;
const int kRealPaddingLeftIdx = 2;
const int kStrideDim = 2;
const int kPaddingDim = 2;
const int kDilationDim = 2;
const int kNCLDIM = 3;
const int kNCHWDIM = 4;
const int kAvgpoolMinH = 10;
const int kAvgpoolMinW = 10;

static const int64_t DIM0 = 0;
static const int64_t DIM1 = 1;
static const int64_t DIM2 = 2;
static const int64_t DIM3 = 3;
static const int64_t DIM4 = 4;

static const int64_t AVGV2_KERNEL_H_MAX = 255;
static const int64_t AVGV2_KERNEL_W_MAX = 255;
static const int64_t AVGV2_KERNEL_SIZE_H_MUL_W = 255;
static const int64_t AVGV2_KERNEL_SIZE = 20;
static const int64_t LONG_STRIP_MULTIPLIER = 2;
template <typename T>
static const aclTensor* GenMeanMatrixIdentity(
    op::DataType dataType, const aclIntArray* meanMatrixSizeNCHW, const aclIntArray* kernelSizeNCHW,
    const aclIntArray* stride2, const aclIntArray* realPadding4, const aclIntArray* inputSizeNCHW,
    aclOpExecutor* executor)
{
    int64_t meanMatrixTotalSize = 1;
    for (size_t i = 0; i < meanMatrixSizeNCHW->Size(); i++) {
        meanMatrixTotalSize *= (*meanMatrixSizeNCHW)[i];
    }
    std::unique_ptr<T[]> input = std::make_unique<T[]>(meanMatrixTotalSize);
    if (input.get() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Fail to request space for MeanMatrix.");
    }
    int64_t addFlagW = 0;
    int64_t addFlagH = 0;
    int64_t offset = 0;
    for (int64_t nc = 0; nc < (*meanMatrixSizeNCHW)[kNDimNCHWIdx] * (*meanMatrixSizeNCHW)[kCDimNCHWIdx]; nc++) {
        for (int64_t h = 0; h < (*meanMatrixSizeNCHW)[kHDimNCHWIdx]; h++) {
            for (int64_t w = 0; w < (*meanMatrixSizeNCHW)[kWDimNCHWIdx]; w++) {
                offset = nc * (*meanMatrixSizeNCHW)[kHDimNCHWIdx] * (*meanMatrixSizeNCHW)[kWDimNCHWIdx] +
                         h * (*meanMatrixSizeNCHW)[kWDimNCHWIdx] + w;
                // 计算kernel窗口里有效数据个数
                for (int64_t hk = 0; hk < (*kernelSizeNCHW)[kHDimNCHWIdx]; hk++) {
                    for (int64_t wk = 0; wk < (*kernelSizeNCHW)[kWDimNCHWIdx]; wk++) {
                        addFlagH = 0;
                        addFlagW = 0;
                        int64_t hIdx = h * (*stride2)[kStrideHIdx] + (hk + 1);
                        int64_t wIdx = w * (*stride2)[kStrideWIdx] + (wk + 1);
                        if ((*realPadding4)[kRealPaddingUpIdx] < hIdx &&
                            hIdx <= (*inputSizeNCHW)[kHDimNCHWIdx] + (*realPadding4)[kRealPaddingUpIdx]) {
                            addFlagH = 1;
                        }
                        if ((*realPadding4)[kRealPaddingLeftIdx] < wIdx &&
                            wIdx <= (*inputSizeNCHW)[kWDimNCHWIdx] + (*realPadding4)[kRealPaddingLeftIdx]) {
                            addFlagW = 1;
                        }
                        if ((addFlagH == 1) && (addFlagW == 1)) {
                            input[offset] += (T)(1);
                        }
                    }
                }
                input[offset] = (T)(1) / input[offset];
            }
        }
    }
    op::Shape meanMatrixShape = op::Shape(
        {(*meanMatrixSizeNCHW)[kNDimNCHWIdx], (*meanMatrixSizeNCHW)[kCDimNCHWIdx], (*meanMatrixSizeNCHW)[kHDimNCHWIdx],
         (*meanMatrixSizeNCHW)[kWDimNCHWIdx]});
    auto meanMatrix = executor->ConvertToTensor(input.get(), meanMatrixTotalSize, dataType);
    CHECK_RET(meanMatrix != nullptr, nullptr);
    const_cast<aclTensor*>(meanMatrix)->SetOriginalShape(meanMatrixShape);
    const_cast<aclTensor*>(meanMatrix)->SetOriginalFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor*>(meanMatrix)->SetStorageShape(meanMatrixShape);
    const_cast<aclTensor*>(meanMatrix)->SetStorageFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor*>(meanMatrix)->SetViewShape(meanMatrixShape);
    const_cast<aclTensor*>(meanMatrix)->SetViewFormat(Format::FORMAT_NCHW);

    return meanMatrix;
}

template <typename T>
static const aclTensor* GenKernelMatrixIdentity(
    op::DataType dataType, const aclIntArray& kernelSizeNCHW, T kernelValue, aclOpExecutor* executor)
{
    int64_t kernelMatrixTotalSize = 1;
    for (size_t i = 0; i < kernelSizeNCHW.Size(); i++) {
        kernelMatrixTotalSize *= (kernelSizeNCHW)[i];
    }
    std::unique_ptr<T[]> input = std::make_unique<T[]>(kernelMatrixTotalSize);
    if (input.get() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Fail to request space for kernelMatrix.");
    }
    for (int64_t i = 0; i < kernelMatrixTotalSize; i++) {
        input[i] = kernelValue;
    }
    op::Shape kernelMatrixShape = op::Shape(
        {kernelSizeNCHW[kNDimNCHWIdx], kernelSizeNCHW[kCDimNCHWIdx], kernelSizeNCHW[kHDimNCHWIdx],
         kernelSizeNCHW[kWDimNCHWIdx]});
    auto kernelMatrixNCHW = executor->ConvertToTensor(input.get(), kernelMatrixTotalSize, dataType);
    CHECK_RET(kernelMatrixNCHW != nullptr, nullptr);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetOriginalShape(kernelMatrixShape);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetOriginalFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetStorageShape(kernelMatrixShape);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetStorageFormat(Format::FORMAT_NCHW);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetViewShape(kernelMatrixShape);
    const_cast<aclTensor*>(kernelMatrixNCHW)->SetViewFormat(Format::FORMAT_NCHW);

    return kernelMatrixNCHW;
}

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static bool CheckNotNull(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* padding,
    aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(kernelSize, return false);
    OP_CHECK_NULL(padding, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return DTYPE_SUPPORT_LIST_WITH_BF16;
    }
    return DTYPE_SUPPORT_LIST;
}

static bool IfSupport2dTo3d()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        return true;
    }
    return false;
}

static bool CheckDtypeVaild(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

    // 当前输入输出tensor的dtype不一致无法支持
    if (gradOutput->GetDataType() != gradInput->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dtype of gradOutput and gradInput should be equal, gradOutput [%s], gradInput [%s].",
            op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(gradInput->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckFormatVaild(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 检查gradOutput的format是否为NCHW/NCL,下同
    if (gradOutput->GetStorageFormat() != Format::FORMAT_NCHW && gradOutput->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "gradOutput format should be NCHW or NCL. Actual: gradOutput is [%s].",
            op::ToString(gradOutput->GetStorageFormat()).GetString());
        return false;
    }

    if (self->GetStorageFormat() != Format::FORMAT_NCHW && self->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "self format should be NCHW or NCL. Actual: self is [%s].",
            op::ToString(self->GetStorageFormat()).GetString());
        return false;
    }

    if (gradInput->GetStorageFormat() != Format::FORMAT_NCHW && gradInput->GetStorageFormat() != Format::FORMAT_NCL) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "gradInput format should be NCHW or NCL. Actual: gradInput is [%s].",
            op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    // 检查当输入是4维时，format是否为NCHW
    if (gradOutput->GetViewShape().GetDimNum() == kNCHWDIM && (gradOutput->GetStorageFormat() != Format::FORMAT_NCHW ||
                                                               gradInput->GetStorageFormat() != Format::FORMAT_NCHW)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "gradOutput and gradInput should all be NCHW when 4D. Actual: gradOutput [%s], gradInput [%s].",
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(gradOutput->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Not support format [%s].",
            op::ToString(gradOutput->GetStorageFormat()).GetString());
        return false;
    }

    return true;
}

static bool CheckDimsVaild(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, const aclTensor* gradInput)
{
    // 校验输入输出维度是否一致
    if (!((gradOutput->GetViewShape().GetDimNum() == kNCLDIM && gradInput->GetViewShape().GetDimNum() == kNCLDIM) ||
          (gradOutput->GetViewShape().GetDimNum() == kNCHWDIM && gradInput->GetViewShape().GetDimNum() == kNCHWDIM))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput and gradInput must be 3D or 4D at the same time.");
        return false;
    }

    // 检查输入输出tensor维度值, 维度值不能小于0
    for (size_t i = 0; i < gradOutput->GetViewShape().GetDimNum(); i++) {
        if (gradOutput->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of gradOutput is negative.");
            return false;
        }
    }

    for (size_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
        if (self->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of self is negative.");
            return false;
        }
    }

    for (size_t i = 0; i < gradInput->GetViewShape().GetDimNum(); i++) {
        if (gradInput->GetViewShape().GetDim(i) < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of gradInput is negative.");
            return false;
        }
    }

    // 检查属性维度,kernelSize、stride和padding维度
    int64_t kernelSizeDim = kernelSize->Size();
    // kernelSize维度不能大于2或者等于0, 维度值必须大于0
    if (kernelSizeDim > 2 || kernelSizeDim == 0) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dims of kernelSize should be 1 or 2, Actual: Dims of kernelSize is [%ld].",
            kernelSizeDim);
        return false;
    }
    for (int64_t i = 0; i < kernelSizeDim; i++) {
        if ((*kernelSize)[i] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of kernelSize is negative or zero.");
            return false;
        }
    }

    if (stride != nullptr) {
        int64_t strideDim = stride->Size();
        // stride维度不能大于2, 维度值必须大于0
        if (strideDim > 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dims of stride should <= 2, Actual: Dims of stride is [%ld].", strideDim);
            return false;
        }
        for (size_t i = 0; i < stride->Size(); i++) {
            if ((*stride)[i] <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value of stride is negative or zero.");
                return false;
            }
        }
    }

    int64_t paddingDim = padding->Size();
    // padding维度不能大于2
    if (paddingDim > 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dims of padding should <= 2, Actual: Dims of padding is [%ld].", paddingDim);
        return false;
    }

    return true;
}

static bool CheckCubeMathTypeValid(int8_t cubeMathType)
{
    if (cubeMathType != KEEP_DTYPE && cubeMathType != ALLOW_FP32_DOWN_PRECISION && cubeMathType != USE_FP16 &&
        cubeMathType != USE_HF32) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "value of cubeMathType cann't be is not in [KEEP_DTYPE, ALLOW_FP32_DOWN_PRECISION, USE_FP16, USE_HF32].");
        return false;
    }
    return true;
}

static bool CheckOutputShape(const aclTensor* self, const aclTensor* gradInput)
{
    if (self->GetViewShape() != gradInput->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "out tensor's shape[%s] is not equel with inferOut shape[%s]",
            op::ToString(self->GetViewShape()).GetString(), op::ToString(gradInput->GetViewShape()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, int8_t cubeMathType, aclTensor* gradInput)
{
    // 检查输入的aclTensor是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, kernelSize, padding, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 检查数据类型是否支持
    CHECK_RET(CheckDtypeVaild(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 检查格式是否支持
    CHECK_RET(CheckFormatVaild(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 检查tensor及属性的维度是否支持
    CHECK_RET(CheckDimsVaild(gradOutput, self, kernelSize, stride, padding, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 针对属性的特殊校验
    // 1、padding不能超过对应位置kernelSize的一半
    CHECK_RET(CheckPaddingValidAvgPool2D(kernelSize, padding), ACLNN_ERR_PARAM_INVALID);

    // cubeMathType需要在支持的范围内
    CHECK_RET(CheckCubeMathTypeValid(cubeMathType), ACLNN_ERR_PARAM_NULLPTR);

    // 对于输出shape是否合理进行check
    CHECK_RET(CheckOutputShape(self, gradInput), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* View3Das4D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCL -> unsqueeze -> reformat -> NCHW
    // unsqueeze input into 4D
    const int64_t appendDim[] = {0};
    aclIntArray* dimUnsqueeze = executor->AllocIntArray(appendDim, 1);
    CHECK_RET(dimUnsqueeze != nullptr, nullptr);
    auto unsqueezedInput = l0op::UnsqueezeNd(input, dimUnsqueeze, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, Format::FORMAT_NCHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* View4Das3D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCHW -> squeeze -> reformat -> NCL
    // squeeze out into 3D
    const int64_t removeDim[] = {0};
    aclIntArray* dimSqueeze = executor->AllocIntArray(removeDim, 1);
    CHECK_RET(dimSqueeze != nullptr, nullptr);
    auto squeezedInput = l0op::SqueezeNd(input, dimSqueeze, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to NCL
    auto reformatInput = l0op::ReFormat(squeezedInput, Format::FORMAT_NCL);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclIntArray* CalculateKernelSizeNCHW(
    const aclTensor* gradOutput, const aclIntArray* kernelSize, aclOpExecutor* executor)
{
    // 计算kernelSize,格式为NCHW
    int64_t kernelH = (*kernelSize)[0];
    int64_t kernelW = kernelSize->Size() == 1 ? (*kernelSize)[0] : (*kernelSize)[1];
    int64_t kernelCout = gradOutput->GetViewShape().GetDimNum() == kNCHWDIM ?
                             gradOutput->GetViewShape().GetDim(kCDimNCHWIdx) :
                             gradOutput->GetViewShape().GetDim(kCDimNCLIdx);
    FVector<int64_t> newKernelSize{kernelCout, 1, kernelH, kernelW};
    const aclIntArray* kernelSizeNCHW = executor->AllocIntArray(newKernelSize.data(), kNCHWDIM);
    return kernelSizeNCHW;
}

static const aclIntArray* CalculateMeanMatrixSizeNCHW(const aclTensor* gradOutput, aclOpExecutor* executor)
{
    // meanMatrix的shape大小等于gradOutput大小, 格式为NCHW
    int64_t outputN = 0;
    int64_t outputC = 0;
    int64_t outputH = 0;
    int64_t outputW = 0;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        outputN = 1;
        outputC = gradOutput->GetViewShape().GetDim(kCDimNCLIdx);
        outputH = gradOutput->GetViewShape().GetDim(kHDimNCLIdx);
        outputW = gradOutput->GetViewShape().GetDim(kWDimNCLIdx);
    } else {
        outputN = gradOutput->GetViewShape().GetDim(kNDimNCHWIdx);
        outputC = gradOutput->GetViewShape().GetDim(kCDimNCHWIdx);
        outputH = gradOutput->GetViewShape().GetDim(kHDimNCHWIdx);
        outputW = gradOutput->GetViewShape().GetDim(kWDimNCHWIdx);
    }
    FVector<int64_t> newMeanMatrixSize{outputN, outputC, outputH, outputW};
    const aclIntArray* meanMatrixSizeNCHW = executor->AllocIntArray(newMeanMatrixSize.data(), kNCHWDIM);
    return meanMatrixSizeNCHW;
}

static const aclIntArray* CalculateStrideHW(
    const aclIntArray* kernelSizeNCHW, const aclIntArray* stride, aclOpExecutor* executor)
{
    // 计算二维stride
    int64_t strideH = 0;
    int64_t strideW = 0;
    if (stride == nullptr) {
        strideH = (*kernelSizeNCHW)[kHDimNCHWIdx];
        strideW = (*kernelSizeNCHW)[kWDimNCHWIdx];
    } else {
        strideH = stride->Size() == 0 ? (*kernelSizeNCHW)[kHDimNCHWIdx] : (*stride)[kStrideHIdx];
        strideW = stride->Size() == 0 ? (*kernelSizeNCHW)[kWDimNCHWIdx] :
                  stride->Size() == 1 ? (*stride)[kStrideHIdx] :
                                        (*stride)[kStrideWIdx];
    }
    FVector<int64_t> newStride{strideH, strideW};
    const aclIntArray* stride2 = executor->AllocIntArray(newStride.data(), kStrideDim);
    return stride2;
}

static const aclIntArray* CalculatePaddingHW(const aclIntArray* padding, aclOpExecutor* executor)
{
    // 计算二维padding
    const int64_t paddingH = (*padding)[kPaddingUpIdx];
    const int64_t paddingW = padding->Size() == 1 ? (*padding)[kPaddingUpIdx] : (*padding)[kPaddingLeftIdx];
    FVector<int64_t> newPadding{paddingH, paddingW};
    const aclIntArray* padding2 = executor->AllocIntArray(newPadding.data(), kPaddingDim);
    return padding2;
}

static const aclIntArray* CalculateInputSizeNCHW(const aclTensor* self, aclOpExecutor* executor)
{
    // 计算inputSize, 格式为NCHW
    int64_t inputN = 0;
    int64_t inputC = 0;
    int64_t inputH = 0;
    int64_t inputW = 0;
    // 处理 NCL 和 NCHW 情况
    if (self->GetViewShape().GetDimNum() == kNCLDIM) {
        inputN = 1;
        inputC = self->GetViewShape().GetDim(kCDimNCLIdx);
        inputH = self->GetViewShape().GetDim(kHDimNCLIdx);
        inputW = self->GetViewShape().GetDim(kWDimNCLIdx);
    } else {
        inputN = self->GetViewShape().GetDim(kNDimNCHWIdx);
        inputC = self->GetViewShape().GetDim(kCDimNCHWIdx);
        inputH = self->GetViewShape().GetDim(kHDimNCHWIdx);
        inputW = self->GetViewShape().GetDim(kWDimNCHWIdx);
    }
    FVector<int64_t> newInputSize{inputN, inputC, inputH, inputW};
    const aclIntArray* inputSizeNCHW = executor->AllocIntArray(newInputSize.data(), kNCHWDIM);
    return inputSizeNCHW;
}

static const aclIntArray* CalculateRealPadding(
    const aclTensor* gradOutput, const aclIntArray* kernelSizeNCHW, const aclIntArray* stride2,
    const aclIntArray* padding2, const aclIntArray* inputSizeNCHW, aclOpExecutor* executor)
{
    int64_t inputH = (*inputSizeNCHW)[kHDimNCHWIdx];
    int64_t inputW = (*inputSizeNCHW)[kWDimNCHWIdx];
    int64_t kernelH = (*kernelSizeNCHW)[kHDimNCHWIdx];
    int64_t kernelW = (*kernelSizeNCHW)[kWDimNCHWIdx];
    int64_t padUp = (*padding2)[kPaddingUpIdx];
    int64_t padLeft = (*padding2)[kPaddingLeftIdx];
    int64_t strideH = (*stride2)[kStrideHIdx];
    int64_t strideW = (*stride2)[kStrideWIdx];
    int64_t outputH = 0;
    int64_t outputW = 0;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        outputH = gradOutput->GetViewShape().GetDim(kHDimNCLIdx);
        outputW = gradOutput->GetViewShape().GetDim(kWDimNCLIdx);
    } else {
        outputH = gradOutput->GetViewShape().GetDim(kHDimNCHWIdx);
        outputW = gradOutput->GetViewShape().GetDim(kWDimNCHWIdx);
    }
    int64_t padsNeededH = std::max(static_cast<int64_t>(0), (outputH - 1) * strideH + kernelH - inputH);
    int64_t padsNeededW = std::max(static_cast<int64_t>(0), (outputW - 1) * strideW + kernelW - inputW);
    int64_t realPaddingUp = padUp;
    int64_t realPaddingDown = padsNeededH - realPaddingUp;
    int64_t realPaddingLeft = padLeft;
    int64_t realPaddingRight = padsNeededW - realPaddingLeft;
    FVector<int64_t> newPadding{realPaddingUp, realPaddingDown, realPaddingLeft, realPaddingRight};
    aclIntArray* realPadding4 = executor->AllocIntArray(newPadding.data(), kNCHWDIM);
    return realPadding4;
}

static bool CheckIsNoNeedMul(bool countIncludePad, int64_t divisorOverride)
{
    // divisorOverride不等于0且在支持范围内 或 countIncludePad=false且输入padding值均为0时,不需要Mul
    if ((divisorOverride < kDivisoroverrideMaxValue && divisorOverride > kDivisoroverrideMinValue &&
         divisorOverride != 0) ||
        countIncludePad) {
        return true;
    }
    // 其余情况都需要插入Mul算子
    return false;
}

static const aclTensor* GenMeanMatrix(
    const aclTensor* gradOutput, const aclIntArray* meanMatrixSizeNCHW, const aclIntArray* kernelSizeNCHW,
    const aclIntArray* stride2, const aclIntArray* realPadding4, const aclIntArray* inputSizeNCHW,
    aclOpExecutor* executor)
{
    auto dataType = gradOutput->GetDataType();
    if (DataType::DT_FLOAT == dataType || DataType::DT_BF16 == dataType) {
        return GenMeanMatrixIdentity<float>(
            DataType::DT_FLOAT, meanMatrixSizeNCHW, kernelSizeNCHW, stride2, realPadding4, inputSizeNCHW, executor);
    }
    return GenMeanMatrixIdentity<fp16_t>(
        dataType, meanMatrixSizeNCHW, kernelSizeNCHW, stride2, realPadding4, inputSizeNCHW, executor);
}

static const aclTensor* GenKernelMatrix(
    const aclTensor* gradOutput, bool isNoNeedMul, int64_t divisorOverride, const aclIntArray* kernelSizeNCHW,
    aclOpExecutor* executor)
{
    // kernelValue初始值是1.0
    float kernelValue = 1.0;
    if (isNoNeedMul) {
        if (divisorOverride != 0) {
            kernelValue = kernelValue / divisorOverride;
        } else {
            // countIncludePad为false, kernelValue默认取1/(kh*kw)
            kernelValue = kernelValue / ((*kernelSizeNCHW)[kHDimNCHWIdx] * (*kernelSizeNCHW)[kWDimNCHWIdx]);
        }
    }
    auto dataType = gradOutput->GetDataType();
    if (dataType == DataType::DT_FLOAT || dataType == DataType::DT_BF16) {
        return GenKernelMatrixIdentity(DataType::DT_FLOAT, *kernelSizeNCHW, kernelValue, executor);
    }
    fp16_t kernelValueFp16 = kernelValue;
    return GenKernelMatrixIdentity(dataType, *kernelSizeNCHW, kernelValueFp16, executor);
}

static bool CheckCubeSupport(int64_t divisorOverride, const aclIntArray* kernelSize)
{
    // divisorOverride不在 [-255,0) && (0,255] 范围内,不支持
    if (!IfSupport2dTo3d()) {
        bool isDivisorOverrideValid =
            (divisorOverride >= kDivisoroverrideMinValue) && (divisorOverride <= kDivisoroverrideMaxValue);
        if (!isDivisorOverrideValid) {
            return false;
        }
    } else {
        const int64_t kH = (*kernelSize)[0];
        const int64_t kW = kernelSize->Size() == 1 ? kH : (*kernelSize)[1];
        bool aiCoreKsize = kH <= AVGV2_KERNEL_H_MAX && kW <= AVGV2_KERNEL_W_MAX;
        bool isSupportKernel =
            (kH * kW <= AVGV2_KERNEL_SIZE_H_MUL_W || (kH <= AVGV2_KERNEL_SIZE && kW <= AVGV2_KERNEL_SIZE));
        bool isDivisorOverrideValid =
            (divisorOverride >= kDivisoroverrideMinValue) && (divisorOverride <= kDivisoroverrideMaxValue);
        if (!isDivisorOverrideValid || !aiCoreKsize || !isSupportKernel) {
            return false;
        }
    }

    return true;
}

static bool CheckNeedChangeTo3D(const aclTensor* gradOutput, const aclTensor* self)
{
    // 当前仅支持h和w大于等于10的场景
    int64_t selfH = 0;
    int64_t selfW = 0;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        selfH = self->GetViewShape().GetDim(kHDimNCLIdx);
        selfW = self->GetViewShape().GetDim(kWDimNCLIdx);
    }
    if (gradOutput->GetViewShape().GetDimNum() == kNCHWDIM) {
        selfH = self->GetViewShape().GetDim(kHDimNCHWIdx);
        selfW = self->GetViewShape().GetDim(kWDimNCHWIdx);
    }
    if (selfH != 0 && selfW != 0) {
        if ((selfW / selfH < LONG_STRIP_MULTIPLIER) && (selfH / selfW < LONG_STRIP_MULTIPLIER)) {
            return false;
        }
    }
    return true;
}

static tuple<const aclIntArray*, const aclIntArray*, const aclIntArray*> FillIntArray(
    const aclIntArray* kernelSize, const aclIntArray* strides, const aclIntArray* paddings, aclOpExecutor* executor)
{
    // 对kernel,stride,pad等参数做扩维
    const int64_t kH = (*kernelSize)[0];
    const int64_t kW = kernelSize->Size() == 1 ? kH : (*kernelSize)[1];
    const int64_t kD = 1;
    FVector<int64_t> newKernel3d{kD, kH, kW};
    const aclIntArray* kernelList = executor->AllocIntArray(newKernel3d.data(), newKernel3d.size());

    const int64_t sH = strides->Size() == 0 ? kH : (*strides)[0];
    const int64_t sW = strides->Size() == 0 ? kW : strides->Size() == 1 ? sH : (*strides)[1];
    const int64_t sD = 1;
    FVector<int64_t> newStride3d{sD, sH, sW};
    const aclIntArray* strideList = executor->AllocIntArray(newStride3d.data(), newStride3d.size());

    const int64_t padH = (*paddings)[0];
    const int64_t padW = paddings->Size() == 1 ? padH : (*paddings)[1];
    const int64_t padD = 0;
    FVector<int64_t> newPad3d{padD, padH, padW};
    const aclIntArray* padList = executor->AllocIntArray(newPad3d.data(), newPad3d.size());

    return std::tuple<const aclIntArray*, const aclIntArray*, const aclIntArray*>(kernelList, strideList, padList);
}

static const aclTensor* CopyShape2OneDim(const aclTensor* self, aclOpExecutor* executor)
{
    FVector<int> shapeVec;
    for (uint64_t i = 0; i < self->GetViewShape().GetDimNum(); i++) {
        shapeVec.push_back(self->GetViewShape().GetDim(i));
    }
    auto shapeCopy = executor->ConvertToTensor(shapeVec.data(), shapeVec.size(), DataType::DT_INT32);
    return shapeCopy;
}

static aclnnStatus BuildAvgPool2dBackwardTo3dGraph(
    const aclTensor* gradOutput, const aclTensor* self, bool ceilMode, bool countIncludePad, int64_t divisorOverride,
    aclTensor* gradInput, aclOpExecutor* executor, const aclIntArray* ksize, const aclIntArray* strides,
    const aclIntArray* pads)
{
    // 先将tensor转为连续性
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3D(NCL)需要先转成4D（NCHW)
    const aclTensor* gradOutput4d = gradOutputContiguous;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        gradOutput4d = View3Das4D(gradOutputContiguous, executor);
    }

    const aclTensor* self4d = selfContiguous;
    if (self->GetViewShape().GetDimNum() == kNCLDIM) {
        self4d = View3Das4D(selfContiguous, executor);
    }
    CHECK_RET(self4d != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // Reshape 4D to 5D NCHW->NC1HW
    auto selfShape = op::ToShapeVector(self4d->GetViewShape());
    FVector<int64_t> selfNewShape = {selfShape[DIM0], selfShape[DIM1], 1, selfShape[DIM2], selfShape[DIM3]};
    aclIntArray* selfshapeArray = executor->AllocIntArray(selfNewShape.data(), selfNewShape.size());
    CHECK_RET(selfshapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto self5d = l0op::Reshape(self4d, selfshapeArray, executor);
    CHECK_RET(self5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputShape = op::ToShapeVector(gradOutput4d->GetViewShape());
    int64_t num = 1;
    int64_t mergeNC = gradOutputShape[DIM0] * gradOutputShape[DIM1];
    FVector<int64_t> gradOutputNewShape = {num, mergeNC, num, gradOutputShape[DIM2], gradOutputShape[DIM3]};
    aclIntArray* gradOutputshapeArray = executor->AllocIntArray(gradOutputNewShape.data(), gradOutputNewShape.size());
    CHECK_RET(gradOutputshapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutput5d = l0op::Reshape(gradOutput4d, gradOutputshapeArray, executor);
    CHECK_RET(gradOutput5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Transpose NCDHW->NDHWC
    FVector<int64_t> transPreDim = {DIM0, DIM2, DIM3, DIM4, DIM1};
    auto permPre = executor->AllocIntArray(transPreDim.data(), transPreDim.size());
    CHECK_RET(permPre != nullptr, ACLNN_ERR_INNER_NULLPTR);
    gradOutput5d = l0op::Transpose(gradOutput5d, permPre, executor);
    CHECK_RET(gradOutput5d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* shapeOrigInput = CopyShape2OneDim(self5d, executor);
    std::string dataFormat = "NDHWC";
    auto avgPool3dBackwardResult = l0op::AvgPool3DGrad(
        self5d, shapeOrigInput, gradOutput5d, ksize, strides, pads, ceilMode, countIncludePad, divisorOverride,
        dataFormat, executor);
    CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // Transpose NDHWC to NCDHW
    FVector<int64_t> transAfterDim = {DIM0, DIM4, DIM1, DIM2, DIM3};
    auto permAfter = executor->AllocIntArray(transAfterDim.data(), transAfterDim.size());
    CHECK_RET(permAfter != nullptr, ACLNN_ERR_INNER_NULLPTR);
    avgPool3dBackwardResult = l0op::Transpose(avgPool3dBackwardResult, permAfter, executor);
    CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Reshape
    auto gradInputShape = op::ToShapeVector(gradInput->GetViewShape());
    auto gradInputShapeArray = executor->AllocIntArray(gradInputShape.data(), gradInputShape.size());
    CHECK_RET(gradInputShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    avgPool3dBackwardResult = l0op::Reshape(avgPool3dBackwardResult, gradInputShapeArray, executor);
    CHECK_RET(avgPool3dBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ViewCopy(avgPool3dBackwardResult, gradInput, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus HandleGradInput(aclTensor* gradInput, const aclTensor* castOut, aclOpExecutor* executor)
{
    if (gradInput->GetViewShape().GetDimNum() == kNCLDIM) {
        const aclTensor* gradInput3d = nullptr;
        gradInput3d = View4Das3D(castOut, executor);
        auto result = l0op::ViewCopy(gradInput3d, gradInput, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    } else {
        auto result = l0op::ViewCopy(castOut, gradInput, executor);
        CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus BuildCubeGraph(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, int64_t divisorOverride, int8_t cubeMathType, aclTensor* gradInput, bool isNoNeedMul,
    aclOpExecutor* executor)
{
    // 准备属性和辅助矩阵, 支持属性不同时是一维或二维
    const aclIntArray* kernelSizeNCHW = CalculateKernelSizeNCHW(gradOutput, kernelSize, executor);
    CHECK_RET(kernelSizeNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // meanMatrixSizeNCHW
    const aclIntArray* meanMatrixSizeNCHW = CalculateMeanMatrixSizeNCHW(gradOutput, executor);
    CHECK_RET(meanMatrixSizeNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // stride2
    const aclIntArray* stride2 = CalculateStrideHW(kernelSizeNCHW, stride, executor);
    CHECK_RET(stride2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // padding2
    const aclIntArray* padding2 = CalculatePaddingHW(padding, executor);
    CHECK_RET(padding2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // dilation 默认为 {1, 1}
    FVector<int64_t> newDilation{1, 1};
    const aclIntArray* dilation2 = executor->AllocIntArray(newDilation.data(), kDilationDim);
    CHECK_RET(dilation2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // inputSizeNCHW
    const aclIntArray* inputSizeNCHW = CalculateInputSizeNCHW(self, executor);
    CHECK_RET(inputSizeNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // realPadding 正向过程中的实际padding
    const aclIntArray* realPadding4 =
        CalculateRealPadding(gradOutput, kernelSizeNCHW, stride2, padding2, inputSizeNCHW, executor);
    CHECK_RET(realPadding4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // groups
    const int groups = self->GetViewShape().GetDimNum() == kNCHWDIM ? self->GetViewShape().GetDim(kCDimNCHWIdx) :
                                                                      self->GetViewShape().GetDim(kCDimNCLIdx);

    // gradOutput --> Contiguous --> 4d/3d --> Cast --> Tranasdata --> l0op --> Tranasdata --> Cast --> 4d/3d -->
    // ViewCopy 先将tensor转为连续性的
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3D(NCL)需要先转成4D(NCHW)
    const aclTensor* gradOutput4d = gradOutputContiguous;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        gradOutput4d = View3Das4D(gradOutputContiguous, executor);
    }
    CHECK_RET(gradOutput4d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* self4d = selfContiguous;
    if (self4d->GetViewShape().GetDimNum() == kNCLDIM) {
        self4d = View3Das4D(self4d, executor);
    }
    CHECK_RET(self4d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Ascend910上仅支持fp16
    const aclTensor* castGradOutput4dTensor = gradOutput4d;
    const aclTensor* castSelf4dTensor = self4d;

    const aclTensor* meanMatrixNCHW = nullptr;
    const aclTensor* kernelMatrixNCHW = nullptr;

    // 构造 meanMatrixNCHW
    if (!isNoNeedMul) {
        meanMatrixNCHW = GenMeanMatrix(
            gradOutput, meanMatrixSizeNCHW, kernelSizeNCHW, stride2, realPadding4, inputSizeNCHW, executor);
        CHECK_RET(meanMatrixNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 构造 kernelMatrixNCHW
    kernelMatrixNCHW = GenKernelMatrix(gradOutput, isNoNeedMul, divisorOverride, kernelSizeNCHW, executor);
    CHECK_RET(kernelMatrixNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果gradOutput和self的数据类型为BF16，需要转换为FLOAT计算
    if (gradOutput->GetDataType() == DataType::DT_BF16 || self->GetDataType() == DataType::DT_BF16) {
        castGradOutput4dTensor = l0op::Cast(gradOutput4d, DataType::DT_FLOAT, executor);
        CHECK_RET(castGradOutput4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        castSelf4dTensor = l0op::Cast(self4d, DataType::DT_FLOAT, executor);
        CHECK_RET(castSelf4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* gradOutputNC1HWC0 = nullptr;
    if (!isNoNeedMul) {
        // 构建的tensor，需从host拷贝至npu，否则会由于输入shape过大出现kernel launch失败问题
        auto deviceMeanMatrixNCHW = op::CopyToNpu(meanMatrixNCHW, executor);
        auto gradOutputND = l0op::Mul(castGradOutput4dTensor, deviceMeanMatrixNCHW, executor);
        CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) {
            gradOutputND = l0op::Cast(gradOutputND, DataType::DT_FLOAT16, executor);
            CHECK_RET(gradOutputND != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        gradOutputNC1HWC0 = gradOutputND;
        // Mul输出不一定是NC1HWC0, 传给conv2dbackpropinput前需转成NC1HWC0
        if (gradOutputND->GetStorageFormat() != Format::FORMAT_NC1HWC0) {
            auto gradOutputNCHW = l0op::ReFormat(gradOutputND, Format::FORMAT_NCHW);
            CHECK_RET(gradOutputNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
            gradOutputNC1HWC0 = l0op::TransData(gradOutputNCHW, Format::FORMAT_NC1HWC0, groups, executor);
            CHECK_RET(gradOutputNC1HWC0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) {
            castGradOutput4dTensor = l0op::Cast(castGradOutput4dTensor, DataType::DT_FLOAT16, executor);
            CHECK_RET(castGradOutput4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        gradOutputNC1HWC0 = l0op::TransData(castGradOutput4dTensor, Format::FORMAT_NC1HWC0, groups, executor);
        CHECK_RET(gradOutputNC1HWC0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) {
        castSelf4dTensor = l0op::Cast(castSelf4dTensor, DataType::DT_FLOAT16, executor);
        CHECK_RET(castSelf4dTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto selfNC1HWC0 = l0op::TransData(castSelf4dTensor, Format::FORMAT_NC1HWC0, groups, executor);
    CHECK_RET(selfNC1HWC0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto deviceKernelMatrixNCHW = op::CopyToNpu(kernelMatrixNCHW, executor);
    if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) {
        deviceKernelMatrixNCHW = l0op::Cast(deviceKernelMatrixNCHW, DataType::DT_FLOAT16, executor);
        CHECK_RET(deviceKernelMatrixNCHW != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto kernelMatrixFractalZ = l0op::TransData(deviceKernelMatrixNCHW, Format::FORMAT_FRACTAL_Z, groups, executor);
    CHECK_RET(kernelMatrixFractalZ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 构造tensor(NC1HWC0格式)和attr
    ConvolutionBackwardInputTensorForAvgPool2d inputTensor = {gradOutputNC1HWC0, selfNC1HWC0, kernelMatrixFractalZ};
    const aclIntArray* biasSize = nullptr;
    const bool transposed = false;
    const aclIntArray* outputPadding = nullptr;
    const aclBoolArray* outputMask = nullptr;
    ConvolutionBackwardParamsForAvgPool2d convolutionBackwardParams = {
        biasSize, stride2, realPadding4, dilation2, transposed, outputPadding, groups, outputMask, cubeMathType};
    auto gradInputNC1HWC0 = CalculateConv2DBackpropInputForAvgPool2d(inputTensor, convolutionBackwardParams, executor);
    CHECK_RET(gradInputNC1HWC0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输出的格式从NC1HWC0转换成NCHW
    auto transGradInput = l0op::TransData(gradInputNC1HWC0, Format::FORMAT_NCHW, groups, executor);
    CHECK_RET(transGradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(transGradInput, gradInput->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return HandleGradInput(gradInput, castOut, executor);
}

static bool IsGlobalAvgPool(
    const aclTensor* gradOutput, const aclTensor* self, int64_t divisorOverride, const aclIntArray* kernelSizeNCHW)
{
    if (divisorOverride != 0) {
        return false;
    }

    // 当前仅支持h和w大于等于10的场景
    int64_t selfH = 0;
    int64_t selfW = 0;

    int64_t gradOutputH = 0;
    int64_t gradOutputW = 0;

    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        selfH = self->GetViewShape().GetDim(kHDimNCLIdx);
        selfW = self->GetViewShape().GetDim(kWDimNCLIdx);
        gradOutputH = gradOutput->GetViewShape().GetDim(kHDimNCLIdx);
        gradOutputW = gradOutput->GetViewShape().GetDim(kWDimNCLIdx);
    }

    if (gradOutput->GetViewShape().GetDimNum() == kNCHWDIM) {
        selfH = self->GetViewShape().GetDim(kHDimNCHWIdx);
        selfW = self->GetViewShape().GetDim(kWDimNCHWIdx);
        gradOutputH = gradOutput->GetViewShape().GetDim(kHDimNCHWIdx);
        gradOutputW = gradOutput->GetViewShape().GetDim(kWDimNCHWIdx);
    }

    // 不支持带stride的场景
    int64_t kernelH = (*kernelSizeNCHW)[kHDimNCHWIdx];
    int64_t kernelW = (*kernelSizeNCHW)[kWDimNCHWIdx];
    if (kernelH != selfH || kernelW != selfW) {
        return false;
    }

    if (selfH < kAvgpoolMinH || selfW < kAvgpoolMinW) {
        return false;
    }

    // global avg pool 是正向结果(N, C, 1, 1)
    if (gradOutputH != 1 || gradOutputW != 1) {
        return false;
    }

    return true;
}

static aclnnStatus BuildGlobalAvgPoolGraph(
    const aclTensor* gradOutput, const aclIntArray* kernelSize, aclTensor* gradInput, int8_t cubeMathType,
    aclOpExecutor* executor)
{
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, executor);
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3D(NCL)需要先转成4D(NCHW)
    const aclTensor* gradOutput4d = gradOutputContiguous;
    if (gradOutput->GetViewShape().GetDimNum() == kNCLDIM) {
        gradOutput4d = View3Das4D(gradOutputContiguous, executor);
    }
    CHECK_RET(gradOutput4d != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* castGradOutput = gradOutput4d;
    if (cubeMathType == ALLOW_FP32_DOWN_PRECISION || cubeMathType == USE_FP16) {
        castGradOutput = l0op::Cast(gradOutput4d, DataType::DT_FLOAT16, executor);
        CHECK_RET(castGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (gradOutput->GetDataType() == DataType::DT_BF16) {
        castGradOutput = l0op::Cast(gradOutput4d, DataType::DT_FLOAT, executor);
        CHECK_RET(castGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 构造kernelMatrix
    int64_t kernelH = (*kernelSize)[0];
    int64_t kernelW = kernelSize->Size() == 1 ? (*kernelSize)[0] : (*kernelSize)[1];
    FVector<int64_t> newKernelSize{1, 1, kernelH, kernelW};
    const aclIntArray* kernelMatrixShape = executor->AllocIntArray(newKernelSize.data(), kNCHWDIM);

    auto dataType = castGradOutput->GetDataType();
    const aclTensor* kernelMatrix = nullptr;
    if (dataType == DataType::DT_FLOAT) {
        float kernelValue = static_cast<float>(kernelH * kernelW);
        kernelMatrix = GenKernelMatrixIdentity(dataType, *kernelMatrixShape, kernelValue, executor);
    } else {
        fp16_t kernelValueFp16 = kernelH * kernelW;
        kernelMatrix = GenKernelMatrixIdentity(dataType, *kernelMatrixShape, kernelValueFp16, executor);
    }
    CHECK_RET(kernelMatrix != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto npuKernelMatrix = op::CopyToNpu(kernelMatrix, executor);
    CHECK_RET(npuKernelMatrix != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto divTensor = l0op::Div(castGradOutput, npuKernelMatrix, executor);
    CHECK_RET(divTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castOut = l0op::Cast(divTensor, gradInput->GetDataType(), executor);
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return HandleGradInput(gradInput, castOut, executor);
}

static aclnnStatus BuildAvgPool2dBackwardGraph(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, int8_t cubeMathType,
    aclTensor* gradInput, aclOpExecutor* executor)
{
    const aclIntArray* kernelSizeNCHW = CalculateKernelSizeNCHW(gradOutput, kernelSize, executor);

    // global avg pool走div算子，提高性能
    bool globalAvgPool = IsGlobalAvgPool(gradOutput, self, divisorOverride, kernelSizeNCHW);
    if (globalAvgPool) {
        auto result = BuildGlobalAvgPoolGraph(gradOutput, kernelSize, gradInput, cubeMathType, executor);
        CHECK_RET(result == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        return ACLNN_SUCCESS;
    }

    // isNoNeedMul
    const bool isNoNeedMul = CheckIsNoNeedMul(countIncludePad, divisorOverride);
    OP_LOGD("Current isNoNeedMul flag is: %d", isNoNeedMul);
    bool isSupport2dTo3d = IfSupport2dTo3d();
    OP_LOGD("Current isSupport2dTo3d flag is: %d", isSupport2dTo3d);

    // 构造计算图 根据芯片不同走不通分支
    if (!isSupport2dTo3d) {
        if (CheckCubeSupport(divisorOverride, kernelSize)) {
            auto ret = BuildCubeGraph(
                gradOutput, self, kernelSize, stride, padding, divisorOverride, cubeMathType, gradInput, isNoNeedMul,
                executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        } else {
            // 当divisorOverride超过支持范围，无法支持
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "divisorOverride only support in [-255, 0) && (0, 255].");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (CheckCubeSupport(divisorOverride, kernelSize) && countIncludePad && !ceilMode &&
            !CheckNeedChangeTo3D(gradOutput, self)) {
            auto ret = BuildCubeGraph(
                gradOutput, self, kernelSize, stride, padding, divisorOverride, cubeMathType, gradInput, isNoNeedMul,
                executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        } else {
            auto intArrayFill = FillIntArray(kernelSize, stride, padding, executor);
            const aclIntArray* ksizeFill = std::get<0>(intArrayFill);
            const aclIntArray* stridesFill = std::get<1>(intArrayFill);
            const aclIntArray* padsFill = std::get<2>(intArrayFill);
            CHECK_RET(ksizeFill != nullptr && stridesFill != nullptr && padsFill != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // 校验数值
            auto arrayRet = CheckArrayDataAvgPoolBackWard(ksizeFill, stridesFill, padsFill);
            CHECK_RET(arrayRet == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

            auto ret = BuildAvgPool2dBackwardTo3dGraph(
                gradOutput, self, ceilMode, countIncludePad, divisorOverride, gradInput, executor, ksizeFill,
                stridesFill, padsFill);
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride,
    const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, int8_t cubeMathType,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnAvgPool2dBackward,
        DFX_IN(gradOutput, self, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride, cubeMathType),
        DFX_OUT(gradInput));

    // 入参检查
    auto ret = CheckParams(gradOutput, self, kernelSize, stride, padding, cubeMathType, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 处理输入输出为空tensor
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 构建AvgPool2dBackward计算图
    ret = BuildAvgPool2dBackwardGraph(
        gradOutput, self, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride, cubeMathType,
        gradInput, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 获取workspace
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAvgPool2dBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAvgPool2dBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
