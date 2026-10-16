/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_logit_grad.h"
#include "logit_grad_l0op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr size_t MAX_LOGIT_GRAD_DIM_NUM = 8;
const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_BF16, DataType::DT_FLOAT16,
                                                            DataType::DT_FLOAT};

bool CheckSocVersion()
{
    switch (GetCurrentPlatformInfo().GetSocVersion()) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND950:
            return true;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LogitGrad only supports Ascend 910B, 910_93 and 950.");
            return false;
    }
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensor* dy, const aclTensor* out)
{
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(dy, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckSocVersion(), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(dy, x->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(x, MAX_LOGIT_GRAD_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL(x, dy, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL(x, out, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(x->GetViewFormat() == Format::FORMAT_ND && x->GetStorageFormat() == Format::FORMAT_ND &&
                 dy->GetViewFormat() == Format::FORMAT_ND && dy->GetStorageFormat() == Format::FORMAT_ND &&
                 out->GetViewFormat() == Format::FORMAT_ND && out->GetStorageFormat() == Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "LogitGrad only supports ND format."), return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnLogitGradGetWorkspaceSize(const aclTensor* x, const aclTensor* dy, double eps, const aclTensor* out,
                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLogitGrad, DFX_IN(x, dy, eps), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(x, dy, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dyContiguous = l0op::Contiguous(dy, uniqueExecutor.get());
    CHECK_RET(dyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const bool needsViewCopy = !IsContiguous(out);
    const aclTensor* kernelOut = nullptr;
    if (needsViewCopy) {
        // The output is write-only; allocate a result without copying the old output values.
        kernelOut = uniqueExecutor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    } else {
        // Preserve the offset and normalize contiguous subview metadata without a data copy.
        kernelOut = l0op::Contiguous(out, uniqueExecutor.get());
    }
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::LogitGrad(xContiguous, dyContiguous, static_cast<float>(eps), kernelOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (needsViewCopy) {
        CHECK_RET(l0op::ViewCopy(result, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogitGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogitGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
