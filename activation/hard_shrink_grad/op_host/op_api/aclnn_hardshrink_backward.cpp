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
 * \file aclnn_hardshrink_backward.cpp
 * \brief
 */

#include "aclnn_hardshrink_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "hardshrink_grad.h"
#include "aclnn_kernels/transdata.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910) {
        return ASCEND910_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
}

inline static bool CheckNotNull(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd, const aclTensor* gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(lambd, return false);
    OP_CHECK_NULL(gradInput, return false);

    return true;
}

inline static bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* /* lambd */, const aclTensor* gradInput)
{
    auto dtypeSupportList = GetDtypeSupportList();
    // 检查gradOutput的数据类型是否在HardShrinkBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, dtypeSupportList, return false);
    // 检查self的数据类型是否在HardShrinkBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    // 检查gradInput的数据类型是否在HardShrinkBackward算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, dtypeSupportList, return false);

    return true;
}

inline static bool CheckPromoteType(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput, op::DataType& promoteType)
{
    // 检查gradOutput和self能否做数据类型推导
    promoteType = op::PromoteType(gradOutput->GetDataType(), self->GetDataType());
    OP_CHECK(
        promoteType != DataType::DT_UNDEFINED,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s and self dtype %s can not promote dtype.",
            op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(self->GetDataType()).GetString()),
        return false);

    // 检查推导后的数据类型是否在算子支持的数据类型之内
    auto dtypeSupportList = GetDtypeSupportList();
    OP_CHECK(
        CheckType(promoteType, dtypeSupportList),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "promote dtype %s should be in dtype support list [%s].",
            op::ToString(promoteType).GetString(), op::ToString(dtypeSupportList).GetString()),
        return false);

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, gradInput->GetDataType(), return false);
    return true;
}

inline static bool CheckShape(const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput)
{
    // 仅check维度是否在8维之内
    OP_CHECK_MAX_DIM(gradOutput, 8, return false);
    OP_CHECK_MAX_DIM(self, 8, return false);
    OP_CHECK_MAX_DIM(gradInput, 8, return false);
    return true;
}

static bool CheckShapeBroadcast(
    const aclTensor* gradOutput, const aclTensor* self, const aclTensor* gradInput, op::Shape& resultShape)
{
    // gradOutput、self的shape必须能够进行broadcast
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(gradOutput, self, resultShape, return false);
    // result shape 与指定的输出gradInput shape 必须相等
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gradInput, resultShape, return false);
    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd, const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, self, lambd, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, self, lambd, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. ND算子不检查数据格式

    // 4. 检查shape是否符合约束
    CHECK_RET(CheckShape(gradOutput, self, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static std::tuple<const aclTensor *, const aclTensor *> BroadcastTo(const aclTensor *gradOutput, const aclTensor *self,
                                                                    aclOpExecutor *executor) {
    Shape broadcastShape;
    if (!BroadcastInferShape(self->GetViewShape(), gradOutput->GetViewShape(), broadcastShape)) {
        return std::tuple(nullptr, nullptr);
    }
    FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = ToShapeVector(broadcastShape);
    auto broadcastShapeArray = executor->AllocIntArray(broadcastDims.data(), broadcastDims.size());
    CHECK_RET(broadcastShapeArray != nullptr, std::tuple(nullptr, nullptr));

    self = l0op::BroadcastTo(self, broadcastShapeArray, executor);
    CHECK_RET(self != nullptr, std::tuple(nullptr, nullptr));

    gradOutput = l0op::BroadcastTo(gradOutput, broadcastShapeArray, executor);
    CHECK_RET(gradOutput != nullptr, std::tuple(nullptr, nullptr));

    return std::tie(gradOutput, self);
}

aclnnStatus aclnnHardshrinkBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclTensor* self, const aclScalar* lambd, aclTensor* gradInput,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHardshrinkBackward, DFX_IN(gradOutput, self, lambd), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, self, lambd, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
    op::DataType promoteType;
    CHECK_RET(CheckPromoteType(gradOutput, self, gradInput, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 空Tensor处理
    if (gradOutput->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 检查能否做shape的broadcast推导
    op::Shape resultShape;
    CHECK_RET(CheckShapeBroadcast(gradOutput, self, gradInput, resultShape), ACLNN_ERR_PARAM_INVALID);

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果是viewShape不同，则需要broadCast成shape相同的
    if (gradOutput->GetViewShape() != self->GetViewShape()) {
        auto broadCastResult = BroadcastTo(gradOutputContiguous, selfContiguous, uniqueExecutor.get());

        gradOutputContiguous = std::get<0>(broadCastResult);
        CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

        selfContiguous = std::get<1>(broadCastResult);
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将输入tensor cast到promote数据类型
    auto gradOutputCasted = l0op::Cast(gradOutputContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子HardShrinkGrad进行计算
    auto lambdFloat = lambd->ToFloat();
    if (lambdFloat < 0) {
        lambdFloat = 0;
    }
    auto hardShrinkGrad = l0op::HardShrinkGrad(gradOutputCasted, selfCasted, lambdFloat, uniqueExecutor.get());
    CHECK_RET(hardShrinkGrad != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果 cast 到gradInput 数据类型
    auto hardShrinkGradCasted = l0op::Cast(hardShrinkGrad, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(hardShrinkGradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(hardShrinkGradCasted, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHardshrinkBackward(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnHardshrinkBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif