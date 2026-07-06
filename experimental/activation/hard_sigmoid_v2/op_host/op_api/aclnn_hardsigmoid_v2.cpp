/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hardsigmoid_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "hard_sigmoid_v2.h"
#include "op_api/aclnn_util.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};

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

static bool CheckNotNull(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInplaceParams(const aclTensor* self)
{
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecHardSigmoidV2GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                     aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto hardSigmoidOpOut = l0op::HardSigmoidV2(selfContiguous, uniqueExecutor.get());
    CHECK_RET(hardSigmoidOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(hardSigmoidOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardsigmoidV2GetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                               aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnHardsigmoidV2, DFX_IN(self), DFX_OUT(out));
    return ExecHardSigmoidV2GetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceHardsigmoidV2GetWorkspaceSize(const aclTensor* self, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnInplaceHardsigmoidV2, DFX_IN(self), DFX_OUT(self));

    auto ret = CheckInplaceParams(self);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto* out = const_cast<aclTensor*>(self);
    return ExecHardSigmoidV2GetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnHardsigmoidV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                               const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHardsigmoidV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceHardsigmoidV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceHardsigmoidV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
