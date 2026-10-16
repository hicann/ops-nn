/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_logit.h"
#include "logit_l0op.h"
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
constexpr size_t MAX_LOGIT_DIM_NUM = 8;
const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_BF16,  DataType::DT_FLOAT16,
                                                            DataType::DT_FLOAT, DataType::DT_INT16,
                                                            DataType::DT_INT8,  DataType::DT_UINT8};

bool CheckSocVersion()
{
    switch (GetCurrentPlatformInfo().GetSocVersion()) {
        case SocVersion::ASCEND910B:
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND950:
            return true;
        default:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Logit only supports Ascend 910B, 910_93 and 950.");
            return false;
    }
}

aclnnStatus CheckParams(const aclTensor* input, const aclTensor* out)
{
    OP_CHECK_NULL(input, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckSocVersion(), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);

    const auto inputDtype = input->GetDataType();
    const auto outDtype = (inputDtype == DataType::DT_INT16 || inputDtype == DataType::DT_INT8 ||
                           inputDtype == DataType::DT_UINT8) ?
                              DataType::DT_FLOAT :
                              inputDtype;
    OP_CHECK_DTYPE_NOT_MATCH(out, outDtype, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(input, MAX_LOGIT_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK(input->GetViewFormat() == Format::FORMAT_ND && input->GetStorageFormat() == Format::FORMAT_ND &&
                 out->GetViewFormat() == Format::FORMAT_ND && out->GetStorageFormat() == Format::FORMAT_ND,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Logit only supports ND format."), return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnLogitGetWorkspaceSize(const aclTensor* input, double eps, const aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnLogit, DFX_IN(input, eps), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Contiguous also normalizes the origin/storage shapes of contiguous subviews without copying data.
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const bool needsViewCopy = !IsContiguous(out);
    const aclTensor* kernelOut = nullptr;
    if (needsViewCopy) {
        // The output is write-only; allocate a result without copying the old output values.
        kernelOut = uniqueExecutor->AllocTensor(out->GetViewShape(), out->GetDataType(), Format::FORMAT_ND);
    } else {
        kernelOut = l0op::Contiguous(out, uniqueExecutor.get());
    }
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::Logit(inputContiguous, static_cast<float>(eps), kernelOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (needsViewCopy) {
        CHECK_RET(l0op::ViewCopy(result, out, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogit(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogit);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
