/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hardsigmoid_backward.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hard_sigmoid_grad_v3.h"
#include "op_api/aclnn_util.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType> &GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    return ASCEND910_DTYPE_SUPPORT_LIST;
}

static inline op::DataType GetComputeType(op::DataType gradType, op::DataType selfType)
{
    auto promoteType = op::PromoteType(gradType, selfType);
    if (promoteType == op::DataType::DT_FLOAT16) {
        return op::DataType::DT_FLOAT;
    }
    return promoteType;
}

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    auto promoteType = GetComputeType(gradOutput->GetDataType(), self->GetDataType());
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, self, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, out, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *out)
{
    CHECK_RET(CheckNotNull(gradOutput, self, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecHardsigmoidBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    auto ret = CheckParams(gradOutput, self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto promoteType = GetComputeType(gradOutput->GetDataType(), self->GetDataType());

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutputContiguousCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguousCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInput = l0op::HardSigmoidGradV3(gradOutputContiguousCasted, selfContiguousCasted, uniqueExecutor.get());
    CHECK_RET(gradInput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradInputCasted = l0op::Cast(gradInput, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(gradInputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradInputCasted, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnHardsigmoidBackward, DFX_IN(gradOutput, self), DFX_OUT(out));
    return ExecHardsigmoidBackwardGetWorkspaceSize(gradOutput, self, out, workspaceSize, executor);
}

aclnnStatus aclnnHardsigmoidBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHardsigmoidBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
