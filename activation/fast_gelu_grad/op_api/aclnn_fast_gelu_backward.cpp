/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_fast_gelu_backward.h"
#include "fast_gelu_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_WITH_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<DataType> NULL_SUPPORT_LIST = {};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (Ops::NN::AclnnUtil::IsRegbase() || GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        return DTYPE_SUPPORT_LIST_WITH_BF16;
    } else {
        return NULL_SUPPORT_LIST;
    }
}

static bool CheckNotNull(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 检查gradOutput和self的dtype是否与gradInput的一致
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gradInput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);

    // 检查gradInput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, GetDtypeSupportList(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 校验gradOutput、self和gradInput的shape是否一致
    OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, gradInput, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);
    // 校验self shape维度是否小于8维
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

static bool CheckFormat(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // self、gradInput的数据格式必须相同
    if (self->GetStorageFormat() != gradInput->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of self and gradInput can't be different, self [%s], gradInput [%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(gradInput->GetStorageFormat()).GetString());
        return false;
    }

    // self、gradOutput的数据格式必须相同
    if (self->GetStorageFormat() != gradOutput->GetStorageFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of self and gradOutput can't be different, self [%s], gradOutput [%s].",
            op::ToString(self->GetStorageFormat()).GetString(),
            op::ToString(gradOutput->GetStorageFormat()).GetString());
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查gradOutput和self的dtype能否作类型推导,类型推导后的结果需要与gradInput一致
    CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查gradOutput和self以及gradInput的维度匹配关系
    CHECK_RET(CheckShape(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查gradOutput和self以及gradInput的format一致
    CHECK_RET(CheckFormat(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastGeluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFastGeluBackward, DFX_IN(gradOutput, self), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty() || gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // gradOutput如果非连续，需要转连续
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子FastGeluGrad进行计算
    auto fastGeluBackwardResult = l0op::FastGeluGrad(gradOutputContiguous, selfContiguous, uniqueExecutor.get());
    CHECK_RET(fastGeluBackwardResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(fastGeluBackwardResult, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFastGeluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFastGeluBackward);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif