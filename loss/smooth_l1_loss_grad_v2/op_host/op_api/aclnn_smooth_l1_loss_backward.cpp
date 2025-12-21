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
 * \file aclnn_smooth_l1_loss_backward.cpp
 * \brief
 */

#include "aclnn_smooth_l1_loss_backward.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "smooth_l1_loss_grad_v2.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_dfx.h"
#include "op_api/level2_base.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

enum class Reduction : int64_t {
    NONE = 0,
    MEAN,
    SUM,
    END
};

static const std::string REDUCTION_STR[] = {"none", "mean", "sum"};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                              DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target,
                         const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOut, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(target, return false);
    OP_CHECK_NULL(gradInput, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target,
                            const aclTensor *gradInput)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
    // 检查gradOut的数据类型是否在SmoothL1LossBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    // 检查self的数据类型是否在SmoothL1LossBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    // 检查target的数据类型是否在SmoothL1LossBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(target, DTYPE_SUPPORT_LIST, return false);
    // 检查gradInput的数据类型是否在SmoothL1LossBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target,
                             const aclTensor *gradInput, op::DataType &promoteType)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
    // 检查gradOut和self能否做数据类型推导
    promoteType = op::PromoteType(gradOut->GetDataType(), self->GetDataType());
    OP_CHECK(promoteType != DataType::DT_UNDEFINED,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOut dtype %s and self dtype %s can not promote dtype.",
                    op::ToString(gradOut->GetDataType()).GetString(), op::ToString(self->GetDataType()).GetString()),
            return false);

    op::DataType savedPromoteType = promoteType;
    promoteType = op::PromoteType(promoteType, target->GetDataType());
    OP_CHECK(promoteType != DataType::DT_UNDEFINED,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOut dtype %s and target dtype %s can not promote dtype.",
                    op::ToString(savedPromoteType).GetString(), op::ToString(target->GetDataType()).GetString()),
            return false);

    // 检查推导后的数据类型是否在算子支持的数据类型之内
    OP_CHECK(CheckType(promoteType, DTYPE_SUPPORT_LIST),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "promote dtype %s should be in dtype support list [%s].",
                    op::ToString(promoteType).GetString(), op::ToString(DTYPE_SUPPORT_LIST).GetString()),
            return false);

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, gradInput->GetDataType(), return false);
    return true;
}

static inline bool CheckShape(const aclTensor* gradOut, const aclTensor* self, const aclTensor* target)
{
    auto gradOutDimNum = gradOut->GetViewShape().GetDimNum();
    auto selfDimNum = self->GetViewShape().GetDimNum();
    auto targetDimNum = target->GetViewShape().GetDimNum();
    size_t maxDim = std::max(std::max(gradOutDimNum, selfDimNum), targetDimNum);
    OP_CHECK(maxDim <= MAX_SUPPORT_DIMS_NUMS,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected tensor must have at most %zu dimensions, but got %zu.",
                     MAX_SUPPORT_DIMS_NUMS, maxDim),
             return false);
    return true;
}

static bool CheckShapeBroadcast(const aclTensor* gradOut, const aclTensor* self, const aclTensor* target,
                                const aclTensor* gradInput, op::Shape &resultShape)
{
    // gradOut、self、 target的shape必须能够进行broadcast
    op::Shape broadcastShapeMid;
    OP_CHECK(BroadcastInferShape(gradOut->GetViewShape(), self->GetViewShape(), broadcastShapeMid),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Broadcast %s and %s failed.",
                     op::ToString(gradOut->GetViewShape()).GetString(), op::ToString(self->GetViewShape()).GetString()),
             return false);
    OP_CHECK(BroadcastInferShape(broadcastShapeMid, target->GetViewShape(), resultShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Broadcast %s and %s failed.",
                     op::ToString(broadcastShapeMid).GetString(), op::ToString(target->GetViewShape()).GetString()),
             return false);
    // result shape 与指定的输出gradInput shape 必须相等
    OP_CHECK(resultShape == gradInput->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Exepected tensor for result to have same size as tensor for gradInput, but %s does not equal %s.",
                     op::ToString(resultShape).GetString(), op::ToString(gradInput->GetViewShape()).GetString()),
             return false);
    return true;
}

static inline bool CheckReduction(int64_t reduction)
{
    OP_CHECK(reduction <= static_cast<int64_t>(Reduction::SUM) && reduction >= static_cast<int64_t>(Reduction::NONE),
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Reduction should be between 0 and 2, but got %ld", reduction),
            return false);
    return true;
}

static inline bool CheckBeta(float beta)
{
    OP_CHECK(beta >= 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "beta should be greater than or equal to 0"),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclTensor* self, const aclTensor* target,
                               int64_t reduction, float beta, const aclTensor *gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOut, self, target, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOut, self, target, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. ND算子不检查数据格式

    // 4. 检查shape是否符合约束
    CHECK_RET(CheckShape(gradOut, self, target), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查reduction参数
    CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查beta
    CHECK_RET(CheckBeta(beta), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckReformat(const aclTensor *input)
{
    input = l0op::ReFormat(input, static_cast<op::Format>(ACL_FORMAT_ND));
    OP_CHECK(input != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input tensor with TransData return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self,
                                                      const aclTensor *target, int64_t reduction,
                                                      float beta, aclTensor *gradInput,
                                                      uint64_t *workspaceSize, aclOpExecutor **executor) {
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnSmoothL1LossBackward, DFX_IN(gradOut, self, target, reduction, beta), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, self, target, reduction, beta, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
    op::DataType promoteType;
    CHECK_RET(CheckPromoteType(gradOut, self, target, gradInput, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 检查能否做shape的broadcast推导
    op::Shape resultShape;
    CHECK_RET(CheckShapeBroadcast(gradOut, self, target, gradInput, resultShape), ACLNN_ERR_PARAM_INVALID);

    // 空Tensor处理
    if (gradOut->IsEmpty() || self->IsEmpty() || target->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOut, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入target转换成连续的tensor
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入tensor cast到promote数据类型
    auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将算子format 统一转变为ND
    ret = CheckReformat(gradOutputCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckReformat(selfCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckReformat(targetCasted);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 调用l0算子SmoothL1LossGradV2进行计算
    auto smoothGrad = l0op::SmoothL1LossGradV2(selfCasted, targetCasted, gradOutputCasted, REDUCTION_STR[reduction],
                                               beta, resultShape, uniqueExecutor.get());
    CHECK_RET(smoothGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果 cast 到gradInput 数据类型
    auto smoothGradCasted = l0op::Cast(smoothGrad, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(smoothGradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(smoothGradCasted, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSmoothL1LossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                      aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSmoothL1LossBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif