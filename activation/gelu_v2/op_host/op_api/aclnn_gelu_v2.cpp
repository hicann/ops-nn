/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_gelu_v2.h"
#include "gelu_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

enum Approximate
{
    NONE = 0,
    TANH = 1
};

const std::string GetApproximateStr(int64_t approximate)
{
    if (approximate == Approximate::TANH) {
        return "tanh";
    } else {
        return "none";
    }
}

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    // x和y数据类型必须一样
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);

    if (!CheckSocVersionIsSupportBf16() && (x->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype of gelu_v2 is not support bfloat16 in current socversion.");
        return false;
    }

    // 检查x的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    // x和y的shape必须一致
    OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluV2GetWorkspaceSize(
    const aclTensor* x, int64_t approximate, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnGeluV2, DFX_IN(x, approximate), DFX_OUT(y));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor处理
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x 如果非连续，需要转连续
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子Gelu进行计算
    const std::string approximateStr = GetApproximateStr(approximate);
    auto geluResult = l0op::GeluV2(xContiguous, approximateStr, uniqueExecutor.get());
    CHECK_RET(geluResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(geluResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGeluV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGeluV2);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif