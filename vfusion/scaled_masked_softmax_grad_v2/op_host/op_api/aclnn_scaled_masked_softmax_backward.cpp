/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_scaled_masked_softmax_backward.h"
#include <dlfcn.h>
#include <acl/acl.h>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/data_type_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
    constexpr int32_t INPUT_DIM_NUM = 4;
    constexpr int32_t D_LIMIT = 4096;
}

extern aclnnStatus aclnnInnerScaledMaskedSoftmaxGradV2GetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, double scale, bool fixTriuMask,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerScaledMaskedSoftmaxGradV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream);

static const std::initializer_list<op::DataType> TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static inline bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, TYPE_SUPPORT_LIST, return false);
    if (mask != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(mask, {op::DataType::DT_BOOL}, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, TYPE_SUPPORT_LIST, return false);
    return true;
}
static inline bool CheckFormat(const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, const aclTensor* out)
{
    bool formatValid = gradOutput->GetStorageFormat() == op::Format::FORMAT_ND &&
                       y->GetStorageFormat() == op::Format::FORMAT_ND &&
                       out->GetStorageFormat() == op::Format::FORMAT_ND;
    if (!formatValid) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input and output format only support [ND]. Actual: gradOutput:[%s], y:[%s], out:[%s].",
            op::ToString(gradOutput->GetStorageFormat()).GetString(), op::ToString(y->GetStorageFormat()).GetString(),
                op::ToString(out->GetStorageFormat()).GetString());
        return false;
    }
    if (mask != nullptr) {
        formatValid = formatValid && mask->GetStorageFormat() == op::Format::FORMAT_ND;
    }
    if (!formatValid) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Mask format only support [ND]. Actual: mask:[%s].",
            op::ToString(mask->GetStorageFormat()).GetString());
        return false;
    }
    return formatValid;
}

static inline bool CheckShape(const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, const aclTensor* out)
{
    if (gradOutput->GetViewShape().GetDimNum() != INPUT_DIM_NUM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "gradOutput only support 4 dims, but current is %lu",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }
    int64_t batch = gradOutput->GetViewShape().GetDim(DIM_0);
    int64_t channel = gradOutput->GetViewShape().GetDim(DIM_1);
    int64_t seqLength = gradOutput->GetViewShape().GetDim(DIM_2);
    int64_t headDim = gradOutput->GetViewShape().GetDim(DIM_3);
    bool isInputVaild = (batch >= 0 && channel >= 0 && seqLength >= 0 && headDim >= 0 && headDim <= D_LIMIT);
    if (!isInputVaild) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "shape of gradOutput is not true, current is %s",
            op::ToString(gradOutput->GetViewShape()).GetString());
        return false;
    }
    if (gradOutput->GetViewShape() != y->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "shape of y should equal to gradOutput, gradOutput is %s, y is %s",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(y->GetViewShape()).GetString());
        return false;
    }
    if (gradOutput->GetViewShape() != out->GetViewShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "shape of out should equal to gradOutput, gradOutput is %s, out is %s",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(out->GetViewShape()).GetString());
        return false;
    }
    if (mask != nullptr) {
        if (mask->GetViewShape().GetDimNum() != INPUT_DIM_NUM) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "mask only support 4 dims, but current is %lu",
                mask->GetViewShape().GetDimNum());
            return false;
        }
        int64_t maskBatch = mask->GetViewShape().GetDim(DIM_0);
        int64_t maskChannel = mask->GetViewShape().GetDim(DIM_1);
        int64_t maskSeqLength = mask->GetViewShape().GetDim(DIM_2);
        int64_t maskHeadDim = mask->GetViewShape().GetDim(DIM_3);
        bool isMaskVaild =
            (maskBatch >= 0 && maskChannel >= 0 && maskSeqLength >= 0 && maskHeadDim >= 0 &&
             (maskSeqLength == seqLength && maskHeadDim == headDim));
        if (!isMaskVaild) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "shape of mask is not true, current is %s",
                op::ToString(mask->GetViewShape()).GetString());
            return false;
        }
        if ((batch != maskBatch && maskBatch != 1) || (channel != maskChannel && maskChannel != 1)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "mask should broadcast to gradOutput, gradOutput is %s, mask is %s",
                op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(mask->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, y, mask, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, y, mask, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format是否符合要求
    CHECK_RET(CheckFormat(gradOutput, y, mask, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否符合要求
    CHECK_RET(CheckShape(gradOutput, y, mask, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* y, const aclTensor* mask, double scale, bool fixTriuMask,
    aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret = CheckParams(gradOutput, y, mask, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (fixTriuMask) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fixTriuMask only support false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto outXGrad = const_cast<aclTensor*>(out);
    outXGrad->SetDataType(gradOutput->GetDataType());

    return aclnnInnerScaledMaskedSoftmaxGradV2GetWorkspaceSize(
        gradOutput, y, mask, scale, false, outXGrad, workspaceSize, executor);
}

aclnnStatus aclnnScaledMaskedSoftmaxBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    return aclnnInnerScaledMaskedSoftmaxGradV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif