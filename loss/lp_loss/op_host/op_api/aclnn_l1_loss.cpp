/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_l1_loss.cpp
 * \brief
 */

#include "aclnn_l1_loss.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/broadcast_to.h"
#include "lp_loss.h"
#include "level0/sub.h"
#include "level0/reduce_sum_op.h"
#include "level0/abs.h"
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
#include "loss/common/level2_base_loss.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static const int64_t REDUCTION_NONE_NUM = 0;
static const int64_t REDUCTION_MEAN_NUM = 1;
static const int64_t REDUCTION_SUM_NUM = 2;
static const int64_t P_LOSS = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_SUPPORT_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> CheckSocVersionIsSupportBf16(void)
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return DTYPE_SUPPORT_LIST_SUPPORT_BF16;
    }
    return DTYPE_SUPPORT_LIST_NOT_SUPPORT_BF16;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* target, const aclTensor* out, int64_t reduction)
{
    // 与Cuda行为保持一致
    // 1. If reduction='none', when dtype of self is no floating point, than target must not be floating point.
    if (reduction == REDUCTION_NONE_NUM) {
        if (!IsFloatingType(self->GetDataType()) && IsFloatingType(target->GetDataType())) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Input dtype should be either floating point or complex dtypes. Got %s instead.",
                ToString(self->GetDataType()).GetString());
            return false;
        }
    }

    // 2. If reduction=='mean', when dtype of self is no floating point, than target must be floating point.
    if (reduction == REDUCTION_MEAN_NUM) {
        if (!IsFloatingType(self->GetDataType()) && !IsFloatingType(target->GetDataType())) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "Input dtype should be either floating point or complex dtypes. Got self %s and target %s instead.",
                ToString(self->GetDataType()).GetString(), ToString(target->GetDataType()).GetString());
            return false;
        }
    }

    // 检查self和target能否做数据类型推导
    op::DataType promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Self dtype %s and target dtype %s can not be promoted.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(target->GetDataType()).GetString());
        return false;
    }

    // 检查promoteType的数据类型是否在支持列表内
    auto dtypeSupportList = CheckSocVersionIsSupportBf16();
    if (!CheckType(promoteType, dtypeSupportList)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "promoteType dtype %s should be in dtype support list %s.",
            op::ToString(promoteType).GetString(), op::ToString(dtypeSupportList).GetString());
        return false;
    }

    // LpLoss算子入参和出参的dtype一致，检查promote后的数据类型能否cast为out的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

// 检查传入的reduction数值是否在可选范围内
static bool CheckReduction(int64_t reduction)
{
    if (reduction > REDUCTION_SUM_NUM || reduction < REDUCTION_NONE_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction to be 0 or 1 or 2, but got %ld.", reduction);
        return false;
    }
    return true;
}

// 检查self、target、out的shape是否满足broadcast规则
static bool CheckShape(const aclTensor* self, const aclTensor* target, const aclTensor* out, int64_t reduction)
{
    // self和target的维度不能超过8
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(target, MAX_DIM_LEN, return false);

    op::Shape broadcastShape;
    // self和target能否做broadcast
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(self, target, broadcastShape, return false);

    // 1、若reduction='none'，out的shape与broadcast的结果一致
    // 2、若reduction！='none'，out为0-dimensional Tensor
    if (reduction == REDUCTION_NONE_NUM) {
        if (broadcastShape != out->GetViewShape()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Expected out tensor shape to be %s, but got %s.",
                op::ToString(broadcastShape).GetString(), op::ToString(out->GetViewShape()).GetString());
            return false;
        }
    }
    if (reduction != REDUCTION_NONE_NUM) {
        if (out->GetViewShape().GetDimNum() >= 1) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Expected a 0-dimensional tensor, but Shape of out is %s.",
                op::ToString(out->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* target, int64_t reduction, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull3Tensor(self, target, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查reduction是否符合规则
    CHECK_RET(CheckReduction(reduction), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, target, out, reduction), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输出shape
    CHECK_RET(CheckShape(self, target, out, reduction), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus L1LossEmptyTensorCompute(int64_t reduction, aclTensor* out, aclOpExecutor* executor)
{
    aclnnStatus ret;
    // 1. reduction='none'，返回空tensor
    // 2. reduction='mean'，返回NAN
    // 3. reduction='sum'，返回0
    if (reduction == REDUCTION_NONE_NUM) {
        return ACLNN_SUCCESS;
    }
    if (reduction == REDUCTION_MEAN_NUM) {
        ret = CheckFillScalarLoss(out, NAN, executor);
    }
    if (reduction == REDUCTION_SUM_NUM) {
        ret = CheckFillScalarLoss(out, 0, executor);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

static const aclIntArray* GetSumDim(const aclTensor* self, aclOpExecutor* executor)
{
    op::Shape shape = self->GetViewShape();
    size_t dimDum = shape.GetDimNum();
    int64_t appendDim[dimDum];
    for (uint64_t i = 0; i < dimDum; i++) {
        appendDim[i] = i;
    }
    return executor->AllocIntArray(appendDim, dimDum);
}

static const aclTensor* GetL1LossFromInt64(
    const aclTensor* self, const aclTensor* target, int64_t reduction, aclOpExecutor* executor)
{
    // 不支持复数，输入为int64场景，需要处理none和sum两种模式
    // 调用Sub算子kernel,alpha为1,直接sub
    auto subOpOut = l0op::Sub(self, target, executor);
    if (subOpOut == nullptr) {
        return nullptr;
    }

    if (reduction == REDUCTION_SUM_NUM) {
        auto subAbs = l0op::Abs(subOpOut, executor);
        if (subAbs == nullptr) {
            return nullptr;
        }
        auto dim = GetSumDim(self, executor);
        if (dim == nullptr) {
            return nullptr;
        }
        // reduceSum支持int64,1980AiCPU,1971AiCore
        return l0op::ReduceSumOp(subAbs, dim, true, executor);
    }

    if (reduction == REDUCTION_NONE_NUM) {
        return l0op::Abs(subOpOut, executor);
    }
    return nullptr;
}

aclnnStatus aclnnL1LossGetWorkspaceSize(
    const aclTensor* self, const aclTensor* target, int64_t reduction, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnL1Loss, DFX_IN(self, target, reduction), DFX_OUT(out));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, target, reduction, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty() || target->IsEmpty()) {
        // 根据实际支持情况补充
        ret = L1LossEmptyTensorCompute(reduction, out, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // L1Loss算子需要对self和target两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(self->GetDataType(), target->GetDataType());

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = selfContiguous;
    if (self->GetDataType() != promoteType) {
        selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将输入target转换成连续的tensor
    auto targetContiguous = l0op::Contiguous(target, uniqueExecutor.get());
    CHECK_RET(targetContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto targetCasted = targetContiguous;
    if (target->GetDataType() != promoteType) {
        targetCasted = l0op::Cast(targetContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(targetCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto selfBroadcast = selfCasted;
    auto targetBroadcast = targetCasted;
    // 判断输入shape不相等需要调用BroadcastTo
    if (self->GetViewShape() != target->GetViewShape()) {
        op::Shape broadcastShape;
        if (BroadcastInferShape(self->GetViewShape(), target->GetViewShape(), broadcastShape)) {
            op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(broadcastShape);
            auto broadcastShapeArray = uniqueExecutor.get()->AllocIntArray(broadcastDims.data(), broadcastDims.size());
            CHECK_RET(broadcastShapeArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
            selfBroadcast = l0op::BroadcastTo(selfCasted, broadcastShapeArray, uniqueExecutor.get());
            CHECK_RET(selfBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
            targetBroadcast = l0op::BroadcastTo(targetCasted, broadcastShapeArray, uniqueExecutor.get());
            CHECK_RET(targetBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }

    // 进行L1loss计算,INT64类型采用小算子拼接
    const aclTensor* lossOut = nullptr;
    if (selfBroadcast->GetDataType() == op::DataType::DT_INT64) {
        // 调用小算子组合
        lossOut = GetL1LossFromInt64(selfBroadcast, targetBroadcast, reduction, uniqueExecutor.get());
        CHECK_RET(lossOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        lossOut = l0op::LpLoss(selfBroadcast, targetBroadcast, P_LOSS, reduction, uniqueExecutor.get());
        CHECK_RET(lossOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = lossOut;
    if (lossOut->GetDataType() != out->GetDataType()) {
        castOut = l0op::Cast(lossOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnL1Loss(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnL1Loss);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
