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
 * \file aclnn_hardswish_backward_v2.cpp
 * \brief
 */
#include "aclnn_hardswish_backward_v2.h"
#include "hard_swish_grad_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* HardswishBackwardV2 算子的完整计算流程如下:
 * self                               other
 *   |                                  |
 *   \                                  /
 * Contiguous(workspace_0)    Contiguous(workspace_2)
 *      \                             /
 *       HardswishBackwardV2(workspace_1)
 *                    |
 *                ViewCopy
 *                    |
 *                  result
 */

 namespace {
    // 根据API定义，需要列出所能支持的所有dtype
    static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
        op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
    static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
        op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

    static bool CheckDtypeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
    {
        // 检查gradOutput的数据类型是否在HardSwishGradV2算子的支持列表内
        auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
        OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
        OP_CHECK_DTYPE_NOT_MATCH(gradOutput, self->GetDataType(), return false);
        OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
        return true;
    }

    static bool CheckShapeValid(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
    {
        OP_CHECK_MAX_DIM(gradOutput, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(gradOutput, self, return false);
        OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
        return true;
    }

    static aclnnStatus CheckParams(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* out)
    {
        // 1. 检查参数是否为空指针
        CHECK_RET(CheckNotNull3Tensor(gradOutput, self, out), ACLNN_ERR_PARAM_NULLPTR);

        // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
        CHECK_RET(CheckDtypeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

        // 3. 检查双输入shape是否符合要求
        CHECK_RET(CheckShapeValid(gradOutput, self, out), ACLNN_ERR_PARAM_INVALID);

        return ACLNN_SUCCESS;
    }
}

aclnnStatus aclnnHardswishBackwardV2GetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHardswishBackwardV2, DFX_IN(gradOutput, self), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // HardSwishGradV2算子的空tensor在kernel中支持
    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutputContiguous转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用HardSwishGradV2算子kernel
    auto hardswishBackwardOpOut = l0op::HardSwishGradV2(gradOutputContiguous, selfContiguous, uniqueExecutor.get());
    CHECK_RET(hardswishBackwardOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(hardswishBackwardOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardswishBackwardV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHardswishBackwardV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif