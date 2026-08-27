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
 * \file aclnn_broadcast_gradient_args.cpp
 * \brief
 */
#include "aclnn_broadcast_gradient_args.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "broadcast_gradient_args.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，列出支持的数据类型
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32, op::DataType::DT_INT64};

// 检查参数非空
inline static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* y1, const aclTensor* y2)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(y1, return false);
    OP_CHECK_NULL(y2, return false);
    return true;
}

// 检查数据类型
inline static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* y1, const aclTensor* y2)
{
    // x1/x2 必须在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, DTYPE_SUPPORT_LIST, return false);
    // x1 和 x2 的 dtype 必须一致
    if (x1->GetDataType() != x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 and x2 must have same data type, but x1 is %s, x2 is %s.",
                ToString(x1->GetDataType()).GetString(), ToString(x2->GetDataType()).GetString());
        return false;
    }
    // y1/y2 的 dtype 必须与 x1/x2 一致
    if (y1->GetDataType() != x1->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 dtype must be same as x1, but x1 is %s, y1 is %s.",
                ToString(x1->GetDataType()).GetString(), ToString(y1->GetDataType()).GetString());
        return false;
    }
    if (y2->GetDataType() != x2->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y2 dtype must be same as x2, but x2 is %s, y2 is %s.",
                ToString(x2->GetDataType()).GetString(), ToString(y2->GetDataType()).GetString());
        return false;
    }
    return true;
}

// 检查shape：x1/x2/y1/y2 必须为1维，且y1/y2容量不小于max(x1_len, x2_len)
inline static bool CheckShape(const aclTensor* x1, const aclTensor* x2, const aclTensor* y1, const aclTensor* y2)
{
    if (x1->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 must be 1D tensor, but got %zuD.", x1->GetViewShape().GetDimNum());
        return false;
    }
    if (x2->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 must be 1D tensor, but got %zuD.", x2->GetViewShape().GetDimNum());
        return false;
    }
    if (y1->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 must be 1D tensor, but got %zuD.", y1->GetViewShape().GetDimNum());
        return false;
    }
    if (y2->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y2 must be 1D tensor, but got %zuD.", y2->GetViewShape().GetDimNum());
        return false;
    }
    int64_t x1Len = x1->GetViewShape().GetDim(0);
    int64_t x2Len = x2->GetViewShape().GetDim(0);
    int64_t maxInputLen = std::max(x1Len, x2Len);
    int64_t y1Len = y1->GetViewShape().GetDim(0);
    int64_t y2Len = y2->GetViewShape().GetDim(0);
    if (y1Len < maxInputLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y1 size must be >= max(x1_len, x2_len), but y1 is %ld, x1 is %ld, x2 is %ld.",
                y1Len, x1Len, x2Len);
        return false;
    }
    if (y2Len < maxInputLen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y2 size must be >= max(x1_len, x2_len), but y2 is %ld, x1 is %ld, x2 is %ld.",
                y2Len, x1Len, x2Len);
        return false;
    }
    return true;
}

// 汇总参数校验
inline static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2, aclTensor* y1, aclTensor* y2)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, y1, y2), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查数据类型
    CHECK_RET(CheckDtypeValid(x1, x2, y1, y2), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape
    CHECK_RET(CheckShape(x1, x2, y1, y2), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBroadcastGradientArgsGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, aclTensor* y1,
                                                       aclTensor* y2, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnBroadcastGradientArgs, DFX_IN(x1, x2), DFX_OUT(y1, y2));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数校验
    auto ret = CheckParams(x1, x2, y1, y2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 仅支持 ascend950（Regbase架构）
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BroadcastGradientArgs only support ascend950.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 将输入转换为连续tensor
    auto x1Contiguous = l0op::Contiguous(x1, uniqueExecutor.get());
    CHECK_RET(x1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x2Contiguous = l0op::Contiguous(x2, uniqueExecutor.get());
    CHECK_RET(x2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 还原y1/y2的StorageShape和OriginalShape为ViewShape
    // 因为输出shape是动态的（OutputShapeDependOnCompute），需要还原后由框架在kernel执行后通过outShapeTensor刷新
    auto y1ViewShape = y1->GetViewShape();
    y1->SetStorageShape(y1ViewShape);
    y1->SetOriginalShape(y1ViewShape);
    auto y2ViewShape = y2->GetViewShape();
    y2->SetStorageShape(y2ViewShape);
    y2->SetOriginalShape(y2ViewShape);

    // 调用l0op层，将算子加入执行队列
    auto opRet = l0op::BroadcastGradientArgs(x1Contiguous, x2Contiguous, y1, y2, uniqueExecutor.get());
    CHECK_RET(opRet == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 将executor转移给用户
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnBroadcastGradientArgs(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnBroadcastGradientArgs);
    // 固定写法，调用框架能力完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
