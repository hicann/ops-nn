/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_elu_backward.h"
#include "elu_grad_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_910 = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SUPPORT_BF16 = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* gradOutput, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    const aclTensor* selfOrResult, const aclTensor* gradInput)
{
    // gradOutput、alpha、scale、inputScale、selfOrResult、gradInput不能为空指针
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(inputScale, return false);
    OP_CHECK_NULL(selfOrResult, return false);
    OP_CHECK_NULL(gradInput, return false);
    return true;
}

static inline bool CheckDtypeValid(
    const aclTensor* gradOutput, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    const aclTensor* selfOrResult, const aclTensor* gradInput)
{
    // 检查gradOutput和selfOrResult数据类型是否一致
    OP_CHECK_DTYPE_NOT_SAME(gradOutput, selfOrResult, return false);

    // 获取芯片类型,判断是1971还是1980
    bool isSupportBf16 =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95);
    const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_CURRENT =
        isSupportBf16 ? DTYPE_SUPPORT_LIST_SUPPORT_BF16 : DTYPE_SUPPORT_LIST_910;

    // 检查gradOutput数据类型是否在EluGradV2算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, DTYPE_SUPPORT_LIST_CURRENT, return false);

    // 检查alpha的数据类型能否转换为FLOAT
    if (!CanCast(alpha->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "alpha dtype %s can not cast to float32.",
            ToString(alpha->GetDataType()).GetString());
        return false;
    }

    // 检查scale的数据类型能否转换为FLOAT
    if (!CanCast(scale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "scale dtype %s can not cast to float32.",
            ToString(scale->GetDataType()).GetString());
        return false;
    }

    // 检查inputScale的数据类型能否转换为FLOAT
    if (!CanCast(inputScale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "inputScale dtype %s can not cast to float32.",
            ToString(inputScale->GetDataType()).GetString());
        return false;
    }

    // 检查gradOutput的数据类型能否转换为gradInput的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(gradOutput->GetDataType(), gradInput->GetDataType(), return false);
    return true;
}

static inline bool CheckShape(const aclTensor* gradOutput, const aclTensor* selfOrResult, const aclTensor* gradInput)
{
    // gradOutput、selfOrResult和gradInput的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(selfOrResult, gradOutput, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradInput, gradOutput, return false);

    // gradOutput的数据维度不能超过8
    OP_CHECK_MAX_DIM(gradOutput, MAX_DIM_LEN, return false);

    return true;
}

static inline bool CheckAttributeValue(const aclScalar* alpha, bool isResult)
{
    // 当isResult为True时，alpha不能小于0
    if (isResult && (alpha->ToFloat() < 0)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "In-place elu backward calculation is triggered with a negative slope which is "
            "not supported. This is caused by calling in-place forward function with a negative slope, please call "
            "out-of-place version instead.");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* gradOutput, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    bool isResult, const aclTensor* selfOrResult, const aclTensor* gradInput)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(gradOutput, alpha, scale, inputScale, selfOrResult, gradInput), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否准确
    CHECK_RET(CheckDtypeValid(gradOutput, alpha, scale, inputScale, selfOrResult, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否一致
    CHECK_RET(CheckShape(gradOutput, selfOrResult, gradInput), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查alpha的取值是否满足要求
    CHECK_RET(CheckAttributeValue(alpha, isResult), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEluBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, const aclScalar* alpha, const aclScalar* scale, const aclScalar* inputScale,
    bool isResult, const aclTensor* selfOrResult, aclTensor* gradInput, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(
        aclnnEluBackward, DFX_IN(gradOutput, alpha, scale, inputScale, isResult, selfOrResult), DFX_OUT(gradInput));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOutput, alpha, scale, inputScale, isResult, selfOrResult, gradInput);
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

    // 固定写法，将输入selfOrResult转换成连续的Tensor
    auto contiguousSelfOrResult = l0op::Contiguous(selfOrResult, uniqueExecutor.get());
    CHECK_RET(contiguousSelfOrResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用EluGradV2算子kernel
    auto eluGradV2Out = l0op::EluGradV2(
        contiguousGradOutput, alpha->ToFloat(), scale->ToFloat(), inputScale->ToFloat(), isResult,
        contiguousSelfOrResult, uniqueExecutor.get());
    CHECK_RET(eluGradV2Out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出gradInput的数据类型
    auto castOut = l0op::Cast(eluGradV2Out, gradInput->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出gradInput上，gradInput可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, gradInput, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEluBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEluBackward);

    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif