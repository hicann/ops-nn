/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_squared_relu.cpp
 * \brief
 */

#include "aclnn_squared_relu.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "squared_relu_op_api.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                       op::DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor* input, const aclTensor* out)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckSocVersionIsSupportBf16(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch);
}

inline static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out)
{
    // 检查input、out的数据类型是否在算子支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);

    // 资料约束：out的数据类型与input保持一致
    if (input->GetDataType() != out->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out dtype [%s] should be the same as input dtype [%s].",
                op::ToString(out->GetDataType()).GetString(), op::ToString(input->GetDataType()).GetString());
        return false;
    }

    // bf16仅在部分芯片上支持
    if (input->GetDataType() == op::DataType::DT_BF16 && !CheckSocVersionIsSupportBf16()) {
        auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input dtype %s is unsupported by the current SOC version [%s].",
                op::ToString(input->GetDataType()).GetString(), op::ToString(socVersion).GetString());
        return false;
    }
    return true;
}

inline static bool CheckShape(const aclTensor* input, const aclTensor* out)
{
    // 资料约束：out的shape与input保持一致
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);

    // 资料约束：输入shape只支持维度1至8维
    OP_CHECK_MAX_DIM(input, 8, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(input, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查input和out的shape是否一致、维度数是否在支持范围内
    CHECK_RET(CheckShape(input, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecSquaredReluGetWorkspaceSize(const aclTensor* input, const aclTensor* out,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 输出指针由调用方提供，为空时结果无法按约定返回错误码，先于创建执行器与空张量处理校验
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor无需启动kernel
    if (input->IsEmpty()) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsContiguous(out)) {
        // out连续：kernel直接写入out，避免中间tensor与回写拷贝的额外内存和开销
        auto squaredReluOut = l0op::SquaredRelu(inputContiguous, out, uniqueExecutor.get());
        CHECK_RET(squaredReluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // out非连续：先写入连续中间tensor，再按out的逻辑视图回写
        auto tmpOut = uniqueExecutor->AllocTensor(input->GetViewShape(), input->GetDataType(), op::Format::FORMAT_ND);
        CHECK_RET(tmpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto squaredReluOut = l0op::SquaredRelu(inputContiguous, tmpOut, uniqueExecutor.get());
        CHECK_RET(squaredReluOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResult = l0op::ViewCopy(squaredReluOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSquaredReluGetWorkspaceSize(const aclTensor* input, const aclTensor* out, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSquaredRelu, DFX_IN(input), DFX_OUT(out));
    return ExecSquaredReluGetWorkspaceSize(input, out, workspaceSize, executor);
}

aclnnStatus aclnnSquaredRelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnSquaredRelu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
