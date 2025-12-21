/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_hardsigmoid.h"
#include "aclnn_kernels/contiguous.h"
#include "hardsigmoid.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT32, op::DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* self)
{
    // 检查self的数据类型是否在hardsigmoid算子的支持列表内
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    return true;
}
static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // self和out的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入和输出的类型、数据格式是否一致
    if (self->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor's dtype[%s] should be same with output's dtype[%s].",
                op::ToString(self->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (self->GetStorageFormat() != out->GetStorageFormat()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input tensor's format[%s] and output's format[%s] should be same.",
                op::ToString(self->GetStorageFormat()).GetString(), op::ToString(out->GetStorageFormat()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardsigmoidGetWorkspaceSize(
    const aclTensor* self, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHardsigmoid, DFX_IN(self), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor支持
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用hardsigmoid进行计算
    auto hardsigmoidOpOut = l0op::HardSigmoid(selfContiguous, uniqueExecutor.get());
    CHECK_RET(hardsigmoidOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(hardsigmoidOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardsigmoid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHardsigmoid);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceHardsigmoidGetWorkspaceSize(
    const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto out = const_cast<aclTensor*>(self);
    return aclnnHardsigmoidGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceHardsigmoid(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceHardsigmoid);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
