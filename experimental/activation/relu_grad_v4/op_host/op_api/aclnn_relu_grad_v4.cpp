/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_relu_grad_v4.h"

#include <cmath>
#include <limits>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "relu_grad_v4.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MAX_DIM_LEN = 8;
constexpr float RELU_THRESHOLD = 0.0f;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT8, op::DataType::DT_UINT8,
    op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8,  op::DataType::DT_UINT8,   op::DataType::DT_INT32};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    // GetSocVersion() is deprecated in op_api; use GetCurNpuArch() to gate bf16 support.
    // Only DAV_2201 (Atlas A2/A3, configured in def.cpp) supports bf16; matches relu_grad_v2.
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool isBf16Support = (curArch == NpuArch::DAV_2201);
    if (isBf16Support || Ops::NN::AclnnUtil::IsRegbase()) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    return ASCEND910_DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclScalar* threshold,
                         const aclTensor* out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(threshold, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    // self is the ReLU mask: only needs to be uint8 (checking against supportList would be redundant).
    OP_CHECK_DTYPE_NOT_MATCH(self, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOutput, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, self, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, out, return false);
    return true;
}

static bool CheckThresholdValue(const aclScalar* threshold)
{
    return std::fabs(threshold->ToFloat() - RELU_THRESHOLD) <= std::numeric_limits<float>::epsilon();
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclScalar* threshold,
                               const aclTensor* out)
{
    CHECK_RET(CheckNotNull(gradOutput, self, threshold, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckThresholdValue(threshold), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecReluGradV4GetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                                  const aclScalar* threshold, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradOutput, self, threshold, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto thresholdBackwardOut = l0op::ReluGradV4(gradOutputContiguous, selfContiguous, uniqueExecutor.get());
    CHECK_RET(thresholdBackwardOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(thresholdBackwardOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnReluGradV4GetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
                                            const aclScalar* threshold, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnReluGradV4, DFX_IN(gradOutput, self, threshold), DFX_OUT(out));
    return ExecReluGradV4GetWorkspaceSize(gradOutput, self, threshold, out, workspaceSize, executor);
}

aclnnStatus aclnnReluGradV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnReluGradV4);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
