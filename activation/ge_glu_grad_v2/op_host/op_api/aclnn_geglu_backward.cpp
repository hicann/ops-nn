/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "aclnn_geglu_backward.h"
#include "geglu_grad_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_DTYPE_SUPPORT_LIST_WITH_BF16 = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16};

static const std::initializer_list<DataType> DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT};

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
        Ops::NN::AclnnUtil::IsRegbase()) {
        return DTYPE_DTYPE_SUPPORT_LIST_WITH_BF16;
    } else {
        return DTYPE_DTYPE_SUPPORT_LIST;
    }
}

// 根据API定义，需要列出所能支持的所有approximate
static const int64_t APPROXIMATE_NONE_NUM = 0;
static const int64_t APPROXIMATE_TANH_NUM = 1;

static inline bool CheckNotNull(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, const aclTensor* gradInput)
{
    // gradOutput、self、gelu、gradInput不能为空指针
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(gelu, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, const aclTensor* gradInput)
{
    // 检查gradOutput的数据类型是否在GeGluGradV2算子的支持列表内
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, supportList, return false);

    // gradOutput、self、gelu和gradInput的数据类型必须一致
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, self, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gelu, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, gradInput, return false);

    return true;
}

static inline bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, const aclTensor* gradInput, int64_t dim)
{
    // gelu和gradOutput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(gelu, gradOutput, return false);

    // gradInput和self的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(gradInput, self, return false);

    // check dim value
    auto selfDimNum = self->GetViewShape().GetDimNum();
    // 当输入张量是0维时，当作1维处理
    if (selfDimNum == 0) {
        selfDimNum = 1;
    }
    int64_t selfDimNum2 = static_cast<int64_t>(selfDimNum);

    // 检查指定dim是否在self的维度范围内
    if (dim >= selfDimNum2 || dim < -selfDimNum2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "expected dim to be in range of [%ld, %ld], but got %ld.", -selfDimNum2,
            selfDimNum2 - 1, dim);
        return false;
    }

    auto newDim = dim < 0 ? selfDimNum2 + dim : dim;
    size_t newDim2 = static_cast<size_t>(newDim);

    // gradOutput的shape中除dim维外，其它维的大小跟self一样，dim维的大小是self的一半
    auto gradOutputDimNum = gradOutput->GetViewShape().GetDimNum();
    if (gradOutputDimNum == selfDimNum) {
        for (size_t i = 0; i < newDim2; ++i) {
            if (gradOutput->GetViewShape().GetDim(i) != self->GetViewShape().GetDim(i)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "except for the last dimension, all other dimensions of gradOutput: %s and self: %s must equal",
                    op::ToString(gradOutput->GetViewShape()).GetString(),
                    op::ToString(self->GetViewShape()).GetString());
                return false;
            }
        }

        if (gradOutput->GetViewShape().GetDim(newDim2) * 2 != self->GetViewShape().GetDim(newDim2)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "the last dimension of gradOutput: %s must be the half of the last dimension of self: %s",
                op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
            return false;
        }

        for (size_t i = newDim2 + 1; i < selfDimNum; ++i) {
            if (gradOutput->GetViewShape().GetDim(i) != self->GetViewShape().GetDim(i)) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "except for the last dimension, all other dimensions of gradOutput: %s and self: %s must equal",
                    op::ToString(gradOutput->GetViewShape()).GetString(),
                    op::ToString(self->GetViewShape()).GetString());
                return false;
            }
        }
    } else {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "gradOutput: %s and self: %s must have same dimensions",
            op::ToString(gradOutput->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckAttributeValue(int64_t approximate)
{
    // check approximate value
    if (approximate < APPROXIMATE_NONE_NUM || approximate > APPROXIMATE_TANH_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected approximate to be 0 or 1, but got %ld.", approximate);
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, const aclTensor* gradInput, int64_t dim,
    int64_t approximate)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, gelu, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否准确
    CHECK_RET(CheckDtypeValid(gradOutput, self, gelu, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否一致
    CHECK_RET(CheckShape(gradOutput, self, gelu, gradInput, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查alpha的取值是否满足要求
    CHECK_RET(CheckAttributeValue(approximate), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus ExecGeGluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    bool activateLeft, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, gelu, gradInput, dim, approximate);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果gradOutput是空tensor，则gradInput也是空tensor，直接返回
    if (gradOutput->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的Tensor
    auto contiguousGradOutput = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(contiguousGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的Tensor
    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入gelu转换成连续的Tensor
    auto contiguousGelu = l0op::Contiguous(gelu, uniqueExecutor.get());
    CHECK_RET(contiguousGelu != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用GeGluGradV2算子kernel
    auto GeGluGradV2Out = l0op::GeGluGradV2(
        contiguousGradOutput, contiguousSelf, contiguousGelu, dim, approximate, activateLeft, uniqueExecutor.get());
    CHECK_RET(GeGluGradV2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(GeGluGradV2Out, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeGluBackward, DFX_IN(gradOutput, self, gelu, dim, approximate), DFX_OUT(gradInput));
    return ExecGeGluBackwardGetWorkspaceSize(
        gradOutput, self, gelu, dim, approximate, false, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gelu, int64_t dim, int64_t approximate,
    bool activateLeft, aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnGeGluV3Backward, DFX_IN(gradOutput, self, gelu, dim, approximate, activateLeft), DFX_OUT(gradInput));
    return ExecGeGluBackwardGetWorkspaceSize(
        gradOutput, self, gelu, dim, approximate, activateLeft, gradInput, workspaceSize, executor);
}

aclnnStatus aclnnGeGluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeGluBackward);

    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnGeGluV3Backward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeGluV3Backward);

    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
