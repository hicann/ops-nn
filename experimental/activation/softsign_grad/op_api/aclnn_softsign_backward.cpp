/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file aclnn_softsign_backward.cpp
 * @brief ACLNN L2 API 实现 - SoftsignGrad 算子
 *
 * 接口名称: aclnnSoftsignBackward
 */

#include "aclnn_softsign_backward.h"
#include "softsign_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

// 支持的数据类型（arch35 / Ascend950 支持 FP16/FP32/BF16）
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* gradients, const aclTensor* features)
{
    return gradients->IsEmpty() || features->IsEmpty();
}

static bool CheckNotNull(const aclTensor* gradients, const aclTensor* features, const aclTensor* out)
{
    OP_CHECK_NULL(gradients, return false);
    OP_CHECK_NULL(features, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradients, const aclTensor* features, const aclTensor* out)
{
    // gradients 和 features 的 dtype 必须一致
    OP_CHECK_DTYPE_NOT_MATCH(gradients, features->GetDataType(), return false);
    // 输出 dtype 与输入一致
    OP_CHECK_DTYPE_NOT_MATCH(out, gradients->GetDataType(), return false);

    if (!IsDtypeSupported(gradients->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SoftsignBackward dtype not supported: dtype=%d. Supported: FLOAT16, FLOAT, BF16.",
                static_cast<int>(gradients->GetDataType()));
        return false;
    }
    return true;
}

static bool CheckFormat(const aclTensor* gradients, const aclTensor* features, const aclTensor* out)
{
    auto format1 = gradients->GetStorageFormat();
    auto format2 = features->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    if (IsPrivateFormat(format1) || IsPrivateFormat(format2) || IsPrivateFormat(formatOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SoftsignBackward private format not supported: gradients=%d, features=%d, out=%d",
                static_cast<int>(format1), static_cast<int>(format2), static_cast<int>(formatOut));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* gradients, const aclTensor* features, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(gradients, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(features, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    // gradients 和 features shape 必须完全一致（REQ-5.3）
    if (gradients->GetViewShape() != features->GetViewShape()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "SoftsignBackward shape mismatch between gradients and features");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradients, const aclTensor* features, const aclTensor* out)
{
    if (!CheckNotNull(gradients, features, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(gradients, features, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(gradients, features, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(gradients, features, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

/**
 * @brief 第一段接口：计算 workspace 大小
 */
extern "C" aclnnStatus aclnnSoftsignBackwardGetWorkspaceSize(
    const aclTensor* gradients,
    const aclTensor* features,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSoftsignBackward, DFX_IN(gradients, features), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradients, features, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(gradients, features)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 确保输入连续
    auto gradientsContiguous = l0op::Contiguous(gradients, uniqueExecutor.get());
    CHECK_RET(gradientsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto featuresContiguous = l0op::Contiguous(features, uniqueExecutor.get());
    CHECK_RET(featuresContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用 L0 API
    const aclTensor* opResult = l0op::SoftsignGrad(gradientsContiguous, featuresContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 处理输出非连续
    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

/**
 * @brief 第二段接口：执行计算
 */
extern "C" aclnnStatus aclnnSoftsignBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSoftsignBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
