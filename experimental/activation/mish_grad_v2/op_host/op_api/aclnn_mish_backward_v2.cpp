/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_mish_backward_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "mish_grad_v2.h"
#include "op_api/aclnn_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                 op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
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

static bool CheckNotNull(const aclTensor* grad, const aclTensor* x, const aclTensor* xGrad)
{
    OP_CHECK_NULL(grad, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(xGrad, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* grad, const aclTensor* x, const aclTensor* xGrad)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(grad, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xGrad, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(grad, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(grad, xGrad->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* grad, const aclTensor* x, const aclTensor* xGrad)
{
    OP_CHECK_MAX_DIM(grad, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(x, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(xGrad, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(grad, x, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(grad, xGrad, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* grad, const aclTensor* x, aclTensor* xGrad)
{
    CHECK_RET(CheckNotNull(grad, x, xGrad), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(grad, x, xGrad), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(grad, x, xGrad), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecMishBackwardV2GetWorkspaceSize(const aclTensor* grad, const aclTensor* x, aclTensor* xGrad,
                                                      uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(grad, x, xGrad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (grad->IsEmpty() || x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradContiguous = l0op::Contiguous(grad, uniqueExecutor.get());
    CHECK_RET(gradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MishGradV2(gradContiguous, xContiguous, nullptr, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, xGrad, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMishBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput,
                                              uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const aclTensor* grad = gradOutput;
    const aclTensor* x = self;
    aclTensor* xGrad = gradInput;
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMishBackward, DFX_IN(grad, x), DFX_OUT(xGrad));
    return ExecMishBackwardV2GetWorkspaceSize(grad, x, xGrad, workspaceSize, executor);
}

aclnnStatus aclnnMishBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                              const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMishBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
