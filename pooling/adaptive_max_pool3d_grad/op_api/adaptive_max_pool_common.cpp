/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_adaptive_max_pool3d_backward.h"
#include "adaptive_max_pool3d_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/unsqueeze.h"
#include "level0/squeeze.h"
#include "level0/max_pool3d_grad_with_argmax.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/framework_op.h"
#include "pooling/pool_3d_common/op_api/pool_3d_util.h"
#include "adaptive_max_pool_common.h"
using namespace op;

namespace AdaptiveMaxPoolCommon {
static const std::initializer_list<DataType> NULL_DTYPE_SUPPORT_LIST = {};
static const std::initializer_list<DataType> GRAD_DTYPE_SUPPORT_LIST = {
    DataType::DT_BF16, DataType::DT_FLOAT16, DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

static const size_t CDHW_DIMS = 4;

static const int D_DIM = -3;

static const int64_t SPATIAL_DIM_NUM = 3;
static const int64_t KERNEL_SIZE_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t STRIDE_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t PADDING_DIM_NUM = SPATIAL_DIM_NUM;
static const int64_t DILATION_DIM_NUM = SPATIAL_DIM_NUM;

bool CheckNotNullPtr(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput)
{
    // gradOutput, self, indices, kernelSize, stride, padding, dilation, gradInput cannot be null pointers
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}
const std::initializer_list<op::DataType> GetDtypeSupportListBySocVersion()
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

bool CheckShapeSame(
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

bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SAME(self, gradOutput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);
    auto dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDICES_DTYPE_SUPPORT_LIST, return false);
    return true;
}

aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* gradInput)
{
    CHECK_RET(CheckNotNullPtr(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtypeValid(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeSame(gradOutput, self, indices, gradInput), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

const aclTensor* selectLevelZeroOperation(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* indices, aclTensor* /* gradInput */,
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
}

