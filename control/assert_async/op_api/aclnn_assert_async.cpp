/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_assert_async.cpp
 * \brief aclnn two-stage interface implementation for AssertAsync (no output tensor).
 */

#include "aclnn_assert_async.h"

#include <cstring>
#include <string>

#include "assert_async.h"
#include "../op_kernel/arch35/assert_async_tiling_data.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> CONDITION_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BOOL,  op::DataType::DT_INT8,   op::DataType::DT_INT16,     op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_UINT8,  op::DataType::DT_FLOAT16,   op::DataType::DT_FLOAT,
    op::DataType::DT_BF16,  op::DataType::DT_DOUBLE, op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128};

static bool CheckParams(const aclTensor* inputCondition, const char* assertMsg)
{
    OP_CHECK_NULL(inputCondition, return false);
    // inputCondition的数据维度不能超过8
    OP_CHECK_MAX_DIM(inputCondition, MAX_DIM_LEN, return false);
    if (inputCondition->Numel() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputCondition must have numel == 1, but got %ld.", inputCondition->Numel());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(inputCondition, CONDITION_DTYPE_SUPPORT_LIST, return false);
    if (assertMsg != nullptr && strlen(assertMsg) >= ASSERT_ASYNC_MAX_MSG_LEN - 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "assertMsg length %zu must be smaller than %u bytes (buffer %u bytes including the terminating '\\0')",
                strlen(assertMsg), ASSERT_ASYNC_MAX_MSG_LEN - 1, ASSERT_ASYNC_MAX_MSG_LEN);
        return false;
    }
    return true;
}
} // namespace

aclnnStatus aclnnAssertAsyncGetWorkspaceSize(const aclTensor* inputCondition, const char* assertMsg,
                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnAssertAsync, DFX_IN(inputCondition, assertMsg), DFX_OUT());

    // Create op executor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Check input parameters (numel == 1, supported dtype, assertMsg length < 255)
    CHECK_RET(CheckParams(inputCondition, assertMsg), ACLNN_ERR_PARAM_INVALID);

    // Non-contiguous -> contiguous (single element is always contiguous; kept as a uniform fallback).
    const aclTensor* inputContiguous = l0op::Contiguous(inputCondition, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Launch L0 op (no output tensor). nullptr assertMsg is treated as "".
    const std::string msg = (assertMsg == nullptr) ? std::string("") : std::string(assertMsg);
    CHECK_RET(l0op::AssertAsync(inputContiguous, msg, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAssertAsync(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAssertAsync);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
