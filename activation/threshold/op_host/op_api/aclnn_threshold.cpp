/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "threshold.h"
#include "aclnn_threshold.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_INT16, op::DataType::DT_INT64,   op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckNotNull(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(threshold, return false);
    OP_CHECK_NULL(value, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self数据类型是否在Threshold算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

    // 检查self的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(self->GetDataType(), out->GetDataType(), return false);

    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);

    // 检查self和out的shape是否保持一致
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, threshold, value, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ExecThresholdGetWorkspaceSize(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 固定写法，参数检查
    auto ret = CheckParams(self, threshold, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* selfCast = selfContiguous;
    if (selfContiguous->GetDataType() == op::DataType::DT_INT16) {
        selfCast = l0op::Cast(selfContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
    }
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用threshold计算
    auto thresholdOut = l0op::Threshold(selfCast, threshold, value, uniqueExecutor.get());
    CHECK_RET(thresholdOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(thresholdOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnThresholdGetWorkspaceSize(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* value, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnThreshold, DFX_IN(self, threshold, value), DFX_OUT(out));
    return ExecThresholdGetWorkspaceSize(self, threshold, value, out, workspaceSize, executor);
}

aclnnStatus aclnnThreshold(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThreshold);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceThresholdGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* threshold, const aclScalar* value, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnInplaceThreshold, DFX_IN(selfRef, threshold, value), DFX_OUT(selfRef));
    return ExecThresholdGetWorkspaceSize(selfRef, threshold, value, selfRef, workspaceSize, executor);
}

aclnnStatus aclnnInplaceThreshold(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceThreshold);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
