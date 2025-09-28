/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_adaptive_max_pool2d_backward.h"
#include "adaptive_max_pool3d_backward.h"
#include "level0/max_pool3d_grad_with_argmax.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "aclnn_kernels/cast.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> NULL_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST = {
    DataType::DT_BF16, DataType::DT_FLOAT16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const size_t NCL_DIMS = 3;
static const size_t CDHW_DIMS = 4;
static const size_t NCDHW_DIMS = 5;

const int W_DIM = -1;
const int H_DIM = -2;
const int D_DIM = -3;
static const int64_t hDimIndexFromLast = 2;
static const int64_t SPATIAL_DIM_NUM = 3;
static const int64_t KERNEL_SIZE_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t STRIDE_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t PADDING_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t DILATION_DIM_NUM = SPATIAL_DIM_NUM;

const int64_t MAX_INT32 = 2147483647;

static bool CheckNotNullPtr(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput)
{
    // gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput cannot be null pointers
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static const std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion()
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return GRAD_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND910: {
            return NULL_DTYPE_SUPPORT_LIST;
        }
        default: {
            return NULL_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool CheckShapeSame(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    op::Shape gradOutputShape = gradOutput->GetViewShape();
    op::Shape selfShape = self->GetViewShape();
    op::Shape indicesShape = indices->GetViewShape();
    op::Shape gradInputShape = gradInput->GetViewShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t gradInputDimNum = gradInputShape.GetDimNum();
    size_t gradOutputDimNum = gradOutputShape.GetDimNum();
    size_t indicesDimNum = indicesShape.GetDimNum();
    if ((selfDimNum != gradInputDimNum) || (selfDimNum != gradOutputDimNum) || (selfDimNum != indicesDimNum)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Dims of self and gradOutput and gradInput and indices should be same, self[%lu], gradOutput[%lu], "
            "gradInput[%lu], indices[%lu].",
            selfDimNum, gradOutputDimNum, gradInputDimNum, indicesDimNum);
        return false;
    }
    for (size_t idx = 0; idx < selfDimNum; idx++) {
        if (gradOutputShape.GetDim(idx) != indicesShape.GetDim(idx)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput dims No.[%lu] must match indices dims.", idx);
            return false;
        }
        if (selfShape.GetDim(idx) != gradInputShape.GetDim(idx)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self dims No.[%lu] must match gradInput dims.", idx);
            return false;
        }
    }
    return true;
}

static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SAME(self, gradOutput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckFormat(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    if ((self->GetStorageFormat() != gradOutput->GetStorageFormat()) ||
        (self->GetStorageFormat() != gradInput->GetStorageFormat()) ||
        (self->GetStorageFormat() != indices->GetStorageFormat())) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Format of self and gradOutput and gradInput and indices should be same, self[%s], gradOutput[%s], "
            "gradInput[%s], indices[%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(gradOutput->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString(),
            op::ToString(indices->GetStorageFormat()).GetString());
        return false;
    }

    if (op::IsPrivateFormat(self->GetStorageFormat()) || op::IsPrivateFormat(gradInput->GetStorageFormat()) ||
        op::IsPrivateFormat(gradOutput->GetStorageFormat()) || op::IsPrivateFormat(indices->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support NCHW or NCL");
        return false;
    }

    if (self->GetViewShape().GetDimNum() != NCL_DIMS && self->GetViewShape().GetDimNum() != CDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for self but got dim num:%zu",
            self->GetViewShape().GetDimNum());
        return false;
    }

    if (gradInput->GetViewShape().GetDimNum() != NCL_DIMS && gradInput->GetViewShape().GetDimNum() != CDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for gradInput but got dim num:%zu",
            gradInput->GetViewShape().GetDimNum());
        return false;
    }

    if (gradOutput->GetViewShape().GetDimNum() != NCL_DIMS && gradOutput->GetViewShape().GetDimNum() != CDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for gradOutput but got dim num:%zu",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }

    if (indices->GetViewShape().GetDimNum() != NCL_DIMS && indices->GetViewShape().GetDimNum() != CDHW_DIMS) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "4D or 5D tensor expected for indices but got dim num:%zu",
            indices->GetViewShape().GetDimNum());
        return false;
    }

    return true;
}

static bool CheckSelfShapeSupport(const aclTensor* self)
{
    const auto& selfShape = self->GetViewShape();
    const auto& selfDimNum = selfShape.GetDimNum();
    const auto& selfDimW = selfShape.GetDim(selfDimNum + W_DIM);
    const auto& selfDimH = selfShape.GetDim(selfDimNum + H_DIM);

    const int64_t selfSize = selfDimW * selfDimH;
    OP_CHECK(
        (selfSize <= MAX_INT32),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The size of self should be less than or equal to 2^32 - 1, but got selfSize:%ld",
            selfSize),
        return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput)
{
    CHECK_RET(CheckNotNullPtr(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeSame(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckSelfShapeSupport(self), ACLNN_ERR_PARAM_INVALID);
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
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCHW);
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
    auto reformatInput = l0op::ReFormat(squeezedInput, op::Format::FORMAT_NCL);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View4Das5D(const aclTensor* input, aclOpExecutor* executor)
{
    // NCWH -> unsqueeze -> reformat -> NCDHW
    // unsqueeze input into 5D
    auto unsqueezedInput = l0op::UnsqueezeNd(input, hDimIndexFromLast, executor);
    CHECK_RET(unsqueezedInput != nullptr, nullptr);
    // reformat to NCDHW
    auto reformatInput = l0op::ReFormat(unsqueezedInput, op::Format::FORMAT_NCDHW);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static inline const aclTensor* View5Das4D(const aclTensor* input, const op::Format& format, aclOpExecutor* executor)
{
    // NCDHW -> squeeze -> reformat -> NCHW
    // squeeze out into 4D
    auto squeezedInput = l0op::SqueezeNd(input, hDimIndexFromLast, executor);
    CHECK_RET(squeezedInput != nullptr, nullptr);
    // reformat to CDHW
    auto reformatInput = l0op::ReFormat(squeezedInput, format);
    CHECK_RET(reformatInput != nullptr, nullptr);

    return reformatInput;
}

static const aclTensor* selectLevelZeroOperation(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, [[maybe_unused]]aclTensor* gradInput,
    aclOpExecutor* executor)
{
    const auto& selfShape = self->GetViewShape();
    const auto& gradOutputShape = gradOutput->GetViewShape();
    const auto& selfDimNum = selfShape.GetDimNum();
    const auto& gradOutputDimNum = gradOutputShape.GetDimNum();

    std::vector<int64_t> osizes(SPATIAL_DIM_NUM);
    std::vector<int64_t> isizes(SPATIAL_DIM_NUM);
    std::vector<int64_t> kernelSize(KERNEL_SIZE_DIM_NUM);
    std::vector<int64_t> stride(STRIDE_DIM_NUM);

    bool useMaxPool3DGradWithArgmax = false;

    for (size_t dim = 0; dim < SPATIAL_DIM_NUM; ++dim) {
        osizes[dim] = gradOutputShape.GetDim(gradOutputDimNum + D_DIM + dim);
        isizes[dim] = selfShape.GetDim(selfDimNum + D_DIM + dim);
    }

    for (size_t dim = 0; dim < SPATIAL_DIM_NUM; ++dim) {
        if (isizes[dim] % osizes[dim] == 0) {
            kernelSize[dim] = isizes[dim] / osizes[dim];
            stride[dim] = isizes[dim] / osizes[dim];
            useMaxPool3DGradWithArgmax = true;
        } else {
            useMaxPool3DGradWithArgmax = false;
            break;
        }
    }

    if (useMaxPool3DGradWithArgmax) {
        aclIntArray* calculatedKernelSize = aclCreateIntArray(kernelSize.data(), KERNEL_SIZE_DIM_NUM);
        aclIntArray* calculatedStride = aclCreateIntArray(stride.data(), STRIDE_DIM_NUM);
        const int64_t paddingData[PADDING_DIM_NUM] = {0, 0, 0};
        const int64_t dilationData[DILATION_DIM_NUM] = {1, 1, 1};
        aclIntArray* padding = aclCreateIntArray(paddingData, PADDING_DIM_NUM);
        aclIntArray* dilation = aclCreateIntArray(dilationData, DILATION_DIM_NUM);
        bool ceilMode = false;

        auto gradInputResult = l0op::MaxPool3DGradWithArgmax(
            gradOutput, self, indices, calculatedKernelSize, calculatedStride, padding, dilation, ceilMode, executor);

        aclDestroyIntArray(calculatedKernelSize);
        aclDestroyIntArray(calculatedStride);
        aclDestroyIntArray(padding);
        aclDestroyIntArray(dilation);

        return gradInputResult;
    } else {
        return l0op::AdaptiveMaxPool3DGrad(gradOutput, self, indices, executor);
    }

    return nullptr;
}

aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaptiveMaxPool2dBackward, DFX_IN(gradOutput, self, indices), DFX_OUT(gradInput));
    OP_LOGD("AdaptiveMaxPool2DGrad: getWorkspaceSize");

    // Create an OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Сhecking parameters
    auto ret = CheckParams(gradOutput, self, indices, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Check whether the tensor is empty (the operator does not support empty tensors)
    if (gradOutput->IsEmpty() || self->IsEmpty() || indices->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Convert the self, gradOutput, indices into consecutive tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 3D(NCL)需要先转成4D（NCHW)
    const bool isSelf3D = self->GetViewShape().GetDimNum() == NCL_DIMS;
    const aclTensor* gradOutput4d = gradOutputContiguous;
    const aclTensor* self4d = selfContiguous;
    const aclTensor* indices4d = indicesContiguous;
    if (isSelf3D) {
        gradOutput4d = View3Das4D(gradOutputContiguous, uniqueExecutor.get());
        self4d = View3Das4D(selfContiguous, uniqueExecutor.get());
        indices4d = View3Das4D(indicesContiguous, uniqueExecutor.get());
    }

    if (DataType::DT_INT64 == indices4d->GetDataType()) {
        indices4d = l0op::Cast(indices4d, DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(indices4d != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // If it's 4D, it needs to be expanded to 5D before calling the AdaptiveMaxPool3DGrad interface.
    const bool isSelf4D = self4d->GetViewShape().GetDimNum() == CDHW_DIMS;

    auto selfUnsqueezed = isSelf4D ? View4Das5D(self4d, uniqueExecutor.get()) :
                                     l0op::ReFormat(self4d, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(selfUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputUnsqueezed = isSelf4D ? View4Das5D(gradOutput4d, uniqueExecutor.get()) :
                                           l0op::ReFormat(gradOutput4d, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(gradOutputUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesUnsqueezed = isSelf4D ? View4Das5D(indices4d, uniqueExecutor.get()) :
                                        l0op::ReFormat(indices4d, op::Format::FORMAT_NCDHW, uniqueExecutor.get());
    CHECK_RET(indicesUnsqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputResult = selectLevelZeroOperation(
        gradOutputUnsqueezed, selfUnsqueezed, indicesUnsqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(gradInputResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const op::Format& dstFormat = gradOutput4d->GetStorageFormat();
    auto gradResultSqueezed = isSelf4D ? View5Das4D(gradInputResult, dstFormat, uniqueExecutor.get()) :
                                         l0op::ReFormat(gradInputResult, dstFormat, uniqueExecutor.get());
    CHECK_RET(gradResultSqueezed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    gradResultSqueezed = isSelf3D ? View4Das3D(gradResultSqueezed, uniqueExecutor.get()) : gradResultSqueezed;
    CHECK_RET(CheckReduceOutShape(gradResultSqueezed, gradInput), ACLNN_ERR_PARAM_INVALID);

    // If the out is a non-consecutive tensor, convert the calculated consecutive tensor to a non-consecutive tensor
    auto viewGradInputCopyResult = l0op::ViewCopy(gradResultSqueezed, gradInput, uniqueExecutor.get());
    CHECK_RET(viewGradInputCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaptiveMaxPool2dBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaptiveMaxPool2dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif