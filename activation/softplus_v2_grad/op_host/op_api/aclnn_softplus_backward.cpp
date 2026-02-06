/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_softplus_backward.h"
#include "softplus_v2_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// gradOutput 和 gradInput 所支持的 Dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

// self 支持的 Dtype
static const std::initializer_list<op::DataType> SELF_ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> SELF_ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT16, op::DataType::DT_INT32, op::DataType::DT_INT64,
    op::DataType::DT_UINT8, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL, op::DataType::DT_BF16};

// SoftplusV2Grad 所支持的 Dtype
static const std::initializer_list<op::DataType> KERNEL_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static const std::initializer_list<DataType>& GetSelfDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return SELF_ASCEND910B_DTYPE_SUPPORT_LIST;
  } else {
    return SELF_ASCEND910_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput) {
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
    auto SELF_DTYPE_SUPPORT_LIST = GetSelfDtypeSupportList();
    // 输入在支持的范围内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, SELF_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, DTYPE_SUPPORT_LIST, return false);

    // 输入若不在 Kernel 支持的范围内（Float, Float16）, 则判断是否能 cast 为 Float
    auto gradOutputDtype = gradOutput->GetDataType();
    auto selfDtype = self->GetDataType();
    if (!CheckType(gradOutputDtype, KERNEL_SUPPORT_LIST) || !CheckType(selfDtype, KERNEL_SUPPORT_LIST)) {
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(gradOutputDtype, op::DataType::DT_FLOAT, return false);
        OP_CHECK_RESULT_DTYPE_CAST_FAILED(selfDtype, op::DataType::DT_FLOAT, return false);
    }
    return true;
}

static bool CheckShape(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput) {
    OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(gradInput, MAX_SUPPORT_DIMS_NUMS, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(self, gradOutput, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, gradInput, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *gradInput) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否合法
    CHECK_RET(CheckDtypeValid(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}


aclnnStatus aclnnSoftplusBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self,
                                          const aclScalar *beta, const aclScalar *threshold, aclTensor *gradInput,
                                          uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSoftplusBackward, DFX_IN(gradOutput, self, beta, threshold), DFX_OUT(gradInput));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 支持空tensor
    if (gradOutput->IsEmpty() || self->IsEmpty() || gradInput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入转连续，并 cast 到 float
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 若不是float或float16，则转为算子所需 float 类型
    auto gradOutputContiguousCasted = gradOutputContiguous;
    auto selfContiguousCasted = selfContiguous;
    auto goDtype = gradOutput->GetDataType();
    auto selftype = self->GetDataType();
    if (!CheckType(goDtype, KERNEL_SUPPORT_LIST) || !CheckType(selftype, KERNEL_SUPPORT_LIST) || goDtype != selftype) {
        gradOutputContiguousCasted = l0op::Cast(gradOutputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(gradOutputContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
        selfContiguousCasted = l0op::Cast(selfContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(selfContiguousCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    float betaVal = beta->ToFloat();
    float thresholdVal = threshold->ToFloat();

    auto SoftplusV2GradOut = l0op::SoftplusV2Grad(gradOutputContiguousCasted, selfContiguousCasted,
                                                  betaVal, thresholdVal, uniqueExecutor.get());
    CHECK_RET(SoftplusV2GradOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto SoftplusV2GradOutCast = l0op::Cast(SoftplusV2GradOut, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(SoftplusV2GradOutCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(SoftplusV2GradOutCast, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftplusBackward(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnSoftplusBackward);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif