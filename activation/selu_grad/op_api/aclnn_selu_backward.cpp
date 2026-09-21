/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_selu_backward.h"

#include "op_api/op_api_def_nn.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "selugrad.h"
#include "aclnn_kernels/transdata.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static inline const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8};

static inline const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_INT8,
    op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static inline bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* result, aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(result, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* result, const aclTensor* gradInput)
{
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, result, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gradInput, return false);

    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(result, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, supportList, return false);
    return true;
}

static inline bool CheckShape(const aclTensor* gradOutput, const aclTensor* result, const aclTensor* gradInput)
{
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, result, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, gradInput, return false);
    return true;
}

static inline bool CheckRank(const aclTensor* gradOutput, const aclTensor* result, const aclTensor* gradInput)
{
    OP_CHECK_MIN_DIM(gradOutput, 1, return false);
    OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(result, 1, return false);
    OP_CHECK_MAX_DIM(result, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MIN_DIM(gradInput, 1, return false);
    OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* result, aclTensor* gradInput)
{
    CHECK_RET(CheckNotNull(gradOutput, result, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(gradOutput, result, gradInput), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckRank(gradOutput, result, gradInput), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(gradOutput, result, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ExecSeluBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* result, aclTensor* gradInput,
                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradOutput, result, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty() || result->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto resultContiguous = l0op::Contiguous(result, uniqueExecutor.get());
    CHECK_RET(resultContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto seluGradOut = l0op::SeluGrad(gradOutputContiguous, resultContiguous, uniqueExecutor.get());
    CHECK_RET(seluGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto castSeluGradOut = l0op::Cast(seluGradOut, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castSeluGradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyOut = l0op::ViewCopy(castSeluGradOut, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSeluBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* result,
                                              aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSeluBackward, DFX_IN(gradOutput, result), DFX_OUT(gradInput));
    return ExecSeluBackwardGetWorkspaceSize(gradOutput, result, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnSeluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSeluBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
