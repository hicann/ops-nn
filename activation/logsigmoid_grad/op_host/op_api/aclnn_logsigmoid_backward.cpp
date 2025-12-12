/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file aclnn_logsigmoid_backward.cpp
 * \brief
 */

#include "aclnn_logsigmoid_backward.h"

#include <cfloat>
#include <climits>

#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/common/op_error_check.h"

#include "logsigmoid_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT};
static const std::initializer_list<DataType> BF16_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *self,
                              const aclTensor *buffer, aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(buffer, return false);
    OP_CHECK_NULL(gradInput, return false);

    return true;
}

static inline const std::initializer_list<op::DataType>& GetInputDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return BF16_DTYPE_SUPPORT_LIST;
  } else {
    return DTYPE_SUPPORT_LIST;
  }
}

static inline const std::initializer_list<op::DataType>& GetOutputDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
      GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
    return BF16_DTYPE_SUPPORT_LIST;
  } else {
    return DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self,
                            [[maybe_unused]] const aclTensor* buffer, aclTensor* gradInput)
{
    auto inputSupportList = GetInputDtypeSupportList();
    auto outputSupportList = GetOutputDtypeSupportList();
    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, outputSupportList, return false);

    // 检查self的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputSupportList, return false);

    // 检查gradOutput, self, buffer和gradInput的dtype是否一致
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gradInput, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, gradInput, return false);

    return true;
}

static bool CheckShapeValid(const aclTensor* gradOutput, const aclTensor* self,
                            [[maybe_unused]] const aclTensor* buffer, aclTensor* gradInput)
{
    // 检查gradOutput, self, buffer和gradInput的shape是否一致
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradOutput, gradInput->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(self, gradInput->GetViewShape(), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self,
                                  const aclTensor* buffer, aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, buffer, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, buffer, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查双输入shape是否符合要求
    CHECK_RET(CheckShapeValid(gradOutput, self, buffer, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSigmoidBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self,
    const aclTensor* buffer, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnLogSigmoidBackward, DFX_IN(gradOutput, self, buffer), DFX_OUT(gradInput));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, buffer, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，支持空tensor
    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = static_cast<uint64_t>(0);
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 调用LogSigmoidGrad算子Kernel
    auto logsigmoidGradOpOut = l0op::LogSigmoidGrad(gradOutputContiguous, selfContiguous, uniqueExecutor.get());
    CHECK_RET(logsigmoidGradOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(logsigmoidGradOpOut, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSigmoidBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnLogSigmoidBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
