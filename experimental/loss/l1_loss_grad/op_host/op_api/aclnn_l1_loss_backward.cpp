/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_l1_loss_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/broadcast_to.h"
#include "l1_loss_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "loss/common/level2_base_loss.h"
#include "op_api/aclnn_util.h"
#include "platform/soc_spec.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* gradInput)
{
    // 检查self、target和gradOutput能否做数据类型推导
    op::DataType promoteType1 = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());
    if (promoteType1 == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s and Self dtype %s can not be promoted.",
            op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(self->GetDataType()).GetString());
        return false;
    }

    op::DataType promoteType2 = op::PromoteType(target->GetDataType(), promoteType1);
    if (promoteType2 == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Target dtype %s, GradOutput dtype %s and Self dtype %s can not be promoted.",
            op::ToString(target->GetDataType()).GetString(), op::ToString(gradOutput->GetDataType()).GetString(),
            op::ToString(self->GetDataType()).GetString());
        return false;
    }

    // 检查promote后的dtype是否与gradInput的一致
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType2, gradInput->GetDataType(), return false);

    // 检查promoteType的数据类型是否在支持列表内
    const auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        (npuArch == NpuArch::DAV_2201) ? ASCEND910B_DTYPE_SUPPORT_LIST : ASCEND910_DTYPE_SUPPORT_LIST;
    if (!CheckType(promoteType2, CURRENT_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "promoteType dtype %s should be in dtype support list [%s].",
            op::ToString(promoteType2).GetString(), op::ToString(CURRENT_DTYPE_SUPPORT_LIST).GetString());
        return false;
    }
    return true;
}

static bool CheckReduction(int64_t reduction)
{
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction to be 0 or 1 or 2, but got %ld.", reduction);
        return false;
    }
    return true;
}

static bool CheckShape(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, const aclTensor* gradInput)
{
    const auto& gradOutputShape = gradOutput->GetViewShape();
    const auto& selfShape = self->GetViewShape();
    const auto& targetShape = target->GetViewShape();
    const auto& gradInputShape = gradInput->GetViewShape();
    // self和target以及gradOutput的维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(gradInput, MAX_DIM_LEN, return false);

    // 检查gradOutput、self、target是否满足broadcast规则
    // self和gradOutput能否做broadcast
    op::Shape broadcastShape1;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, gradOutput, broadcastShape1, return false);

    // broadcast后的shape能否与target做broadcast
    op::Shape broadcastShape2;
    if (!BroadcastInferShape(targetShape, broadcastShape1, broadcastShape2)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "GradOutput tensor shape:%s self tensor shape:%s target tensor shape:%s can't broadcast.",
            op::ToString(gradOutputShape).GetString(), op::ToString(selfShape).GetString(),
            op::ToString(targetShape).GetString());
        return false;
    }
    if (broadcastShape2 != gradInputShape) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Shape of out should be %s, but current is %s.",
            op::ToString(broadcastShape2).GetString(), op::ToString(gradInputShape).GetString());
        return false;
    }
    return true;
}

static void CheckFormat(const aclTensor* x)
{
    op::Format format = x->GetStorageFormat();
    if (format == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW(
            "Format of input gets [%s], this format mat lead to precision failure", op::ToString(format).GetString());
    }
}

inline aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction,
    const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull4Tensor(gradOutput, self, target, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查reduction是否符合规则
    CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, target, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(CheckShape(gradOutput, self, target, gradInput), ACLNN_ERR_PARAM_INVALID);
    CheckFormat(gradOutput);
    return ACLNN_SUCCESS;
}

// 如果gradOutput、target或者self的shape与braodcast后的shape不一致，在进行反向计算前，先进行broadcasto操作。
static const aclTensor* BroadcastTensor(const aclTensor* self, const op::Shape broadcastShape, aclOpExecutor* executor)
{
    // 如果self的shape与broadcast的不一致，进行BroadcastTo
    if (self->GetViewShape() != broadcastShape) {
        auto broadcastShapeIntArray = GetBroadcastShapeLossBackward(broadcastShape, executor);
        if (broadcastShapeIntArray != nullptr) {
            return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
        }
    }
    return self;
}

aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* target, int64_t reduction,
    aclTensor* gradInput, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnL1LossBackward, DFX_IN(gradOutput, self, target, reduction), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, target, reduction, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty() || gradOutput->IsEmpty() || target->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入target转换成连续的tensor
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // shape不一致进行broadcast
    op::Shape broadcastShape1;
    op::Shape broadcastShape2;
    BroadcastInferShape(gradOutput->GetViewShape(), self->GetViewShape(), broadcastShape1);
    BroadcastInferShape(target->GetViewShape(), broadcastShape1, broadcastShape2);
    auto gradOutputBroadcast = gradOutputContiguous;
    if (!Ops::NN::AclnnUtil::IsRegbase() || !gradOutput->GetViewShape().IsScalar()) {
        // 判断gradOutput是否需要进行broadcast或者promote
        gradOutputBroadcast = BroadcastTensor(gradOutputContiguous, broadcastShape2, uniqueExecutor.get());
        CHECK_RET(gradOutputBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    // 判断gradOutput是否需要进行broadcast
    auto selfBroadcast = BroadcastTensor(selfContiguous, broadcastShape2, uniqueExecutor.get());
    CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 判断gradOutput是否需要进行broadcast
    auto targetBroadcast = BroadcastTensor(targetContiguous, broadcastShape2, uniqueExecutor.get());
    CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对gradOutput、self、target做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType1 = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());
    auto promoteType2 = op::PromoteType(promoteType1, target->GetDataType());
    auto selfCasted = selfBroadcast;
    auto gradOutputCasted = gradOutputBroadcast;
    auto targetCasted = targetBroadcast;

    // 将输入gradOutput、self、target的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    if (promoteType2 != self->GetDataType()) {
        selfCasted = l0op::Cast(selfBroadcast, promoteType2, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (promoteType2 != gradOutput->GetDataType()) {
        gradOutputCasted = l0op::Cast(gradOutputBroadcast, promoteType2, uniqueExecutor.get());
        CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (promoteType2 != target->GetDataType()) {
        targetCasted = l0op::Cast(targetBroadcast, promoteType2, uniqueExecutor.get());
        CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 进行计算
    auto grad = l0op::L1LossGrad(gradOutputCasted, selfCasted, targetCasted, reduction, uniqueExecutor.get());
    CHECK_RET(grad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = grad;
    if (grad->GetDataType() != gradInput->GetDataType()) {
        castOut = l0op::Cast(grad, gradInput->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnL1LossBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnL1LossBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
