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
 * \file aclnn_sigmoid_backward.cpp
 * \brief
 */

#include "aclnn_sigmoid_backward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "sigmoid_grad.h"
#include "opdev/op_dfx.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BF16};

inline static bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

inline static bool CheckNotNull(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(output, return false);
    OP_CHECK_NULL(gradInput, return false);

    return true;
}

inline static bool CheckDtypeValid(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput)
{
    // 检查gradOutput的数据类型是否在sigmoidGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST, return false);
    // 检查output的数据类型是否在sigmoidGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(output, DTYPE_SUPPORT_LIST, return false);
    // 检查gradInput的数据类型是否在sigmoidGrad算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradInput, DTYPE_SUPPORT_LIST, return false);

    bool bf16flag = CheckSocVersionIsSupportBf16();
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (!bf16flag && gradOutput->GetDataType() == op::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s is unsupported by the current SOC version [%s].",
            op::ToString(gradOutput->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

inline static bool CheckShape(const aclTensor *gradOutput, const aclTensor *output,
                                [[maybe_unused]] const aclTensor *gradInput)
{
    // gradOutput 维度必须小于9
    OP_CHECK_MAX_DIM(gradOutput, 8, return false);
    OP_CHECK_MAX_DIM(output, 8, return false);
    return true;
}

static bool CheckPromoteType(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput,
                             op::DataType &promoteType)
{
    // 检查gradOutput和output能否做数据类型推导
    promoteType = op::PromoteType(gradOutput->GetDataType(), output->GetDataType());
    OP_CHECK(promoteType != DataType::DT_UNDEFINED,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GradOutput dtype %s and output dtype %s can not promote dtype.",
                    op::ToString(gradOutput->GetDataType()).GetString(),
                    op::ToString(output->GetDataType()).GetString()),
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

static bool CheckShapeBroadcast(const aclTensor* gradOutput, const aclTensor* output, const aclTensor* gradInput,
                                op::Shape &resultShape)
{
    // gradOutput、output的shape必须能够进行broadcast
    OP_CHECK(BroadcastInferShape(gradOutput->GetViewShape(), output->GetViewShape(), resultShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "the size of tensor self %s must match the size of tensor other %s.",
                     op::ToString(gradOutput->GetViewShape()).GetString(),
                     op::ToString(output->GetViewShape()).GetString()),
             return false);
    // result shape 与指定的输出gradInput shape 必须相等
    OP_CHECK(resultShape == gradInput->GetViewShape(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Exepected tensor for result to have same size as tensor for gradInput, but %s does not equal %s.",
                     op::ToString(resultShape).GetString(), op::ToString(gradInput->GetViewShape()).GetString()),
             return false);
    return true;
}

static aclTensor* BroadcastTensor(const op::Shape dstShape, const aclTensor* src, aclOpExecutor* executor)
{
    auto dstTensor = executor->AllocTensor(dstShape, src->GetDataType());
    op::FVector<int64_t, op::MAX_DIM_NUM> broadcastDims = op::ToShapeVector(dstShape);
    auto shape = executor->ConvertToTensor(broadcastDims.data(), broadcastDims.size(),
        static_cast<op::DataType>(ACL_INT64));
    auto result = l0op::BroadcastTo(src, dstTensor, shape, executor);
    return const_cast<aclTensor*>(result);
}

static aclnnStatus CheckParams(const aclTensor *gradOutput, const aclTensor *output, const aclTensor *gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, output, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(gradOutput, output, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(gradOutput, output, gradInput), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output,
                                                 aclTensor *gradInput, uint64_t *workspaceSize,
                                                 aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnSigmoidBackward, DFX_IN(gradOutput, output), DFX_OUT(gradInput));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, output, gradInput);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查能否做dtype指定的数据类型推导以及推导的数据类型能否转换为输出数据类型
    op::DataType promoteType;
    CHECK_RET(CheckPromoteType(gradOutput, output, gradInput, promoteType), ACLNN_ERR_PARAM_INVALID);

    // 检查能否做shape的broadcast推导
    op::Shape resultShape;
    CHECK_RET(CheckShapeBroadcast(gradOutput, output, gradInput, resultShape), ACLNN_ERR_PARAM_INVALID);

    // sigmoidGrad算子的空tensor在kernel中不支持，对标竞品根据算子实际情况补充
    if (gradOutput->IsEmpty() || output->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入gradOutput转换成连续的tensor
    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入output转换成连续的tensor
    auto outputContiguous = l0op::Contiguous(output, uniqueExecutor.get());
    CHECK_RET(outputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 若输出tensor形状不同，则进行broadcast
    const aclTensor *gradOutputBroadCast = gradOutputContiguous;
    const aclTensor *outputBroadCast = outputContiguous;
    if (gradOutputContiguous->GetViewShape() != outputContiguous->GetViewShape()) {
        gradOutputBroadCast = BroadcastTensor(resultShape, gradOutputContiguous, uniqueExecutor.get());
        CHECK_RET(gradOutputBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        outputBroadCast = BroadcastTensor(resultShape, outputContiguous, uniqueExecutor.get());
        CHECK_RET(outputBroadCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 将输入tensor cast到promote数据类型
    auto gradOutputCast = l0op::Cast(gradOutputBroadCast, promoteType, uniqueExecutor.get());
    CHECK_RET(gradOutputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputCast = l0op::Cast(outputBroadCast, promoteType, uniqueExecutor.get());
    CHECK_RET(outputCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用SigmoidGrad算子Kernel
    auto sigmoidGradOpOut = l0op::SigmoidGrad(outputCast, gradOutputCast, uniqueExecutor.get());
    CHECK_RET(sigmoidGradOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将计算结果 cast 到gradInput 数据类型
    auto sigmoidGradCasted = l0op::Cast(sigmoidGradOpOut, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(sigmoidGradCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(sigmoidGradCasted, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSigmoidBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSigmoidBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif