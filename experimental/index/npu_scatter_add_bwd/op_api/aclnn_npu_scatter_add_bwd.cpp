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
 * \file aclnn_npu_scatter_add_bwd.cpp
 * \brief aclnnNpuScatterAddBwd 两段式接口实现
 */
#include "aclnn_npu_scatter_add_bwd.h"
#include "npu_scatter_add_bwd.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t EXPECTED_Y_GRAD_DIM_NUM = 2;
static constexpr size_t EXPECTED_X_DIM_NUM = 2;
static constexpr size_t EXPECTED_INDEX_DIM_NUM = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* yGrad, const aclTensor* x, const aclTensor* s,
                                const aclTensor* indices, const aclTensor* xGrad, const aclTensor* sGrad)
{
    OP_CHECK_NULL(yGrad, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(s, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(xGrad, return false);
    OP_CHECK_NULL(sGrad, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* yGrad, const aclTensor* x, const aclTensor* s,
                                   const aclTensor* indices, const aclTensor* xGrad, const aclTensor* sGrad)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(yGrad, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, yGrad->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(s, yGrad->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(xGrad, x->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(sGrad, s->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* yGrad, const aclTensor* x, const aclTensor* s, const aclTensor* indices,
                       const aclTensor* xGrad, const aclTensor* sGrad)
{
    OP_CHECK(yGrad->GetViewShape().GetDimNum() == EXPECTED_Y_GRAD_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y_grad should be a 2d tensor, but got %zu dims.",
                     yGrad->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK(
        x->GetViewShape().GetDimNum() == EXPECTED_X_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x should be a 2d tensor, but got %zu dims.", x->GetViewShape().GetDimNum()),
        return false);
    OP_CHECK(
        s->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "s should be a 1d tensor, but got %zu dims.", s->GetViewShape().GetDimNum()),
        return false);
    OP_CHECK(indices->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices should be a 1d tensor, but got %zu dims.",
                     indices->GetViewShape().GetDimNum()),
             return false);

    OP_CHECK(yGrad->GetViewShape().GetDim(1) == x->GetViewShape().GetDim(1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y_grad's dim[1](%ld) and x's dim[1](%ld) should be the same.",
                     yGrad->GetViewShape().GetDim(1), x->GetViewShape().GetDim(1)),
             return false);
    OP_CHECK(x->GetViewShape().GetDim(0) == s->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim[0](%ld) and s's dim[0](%ld) should be the same.",
                     x->GetViewShape().GetDim(0), s->GetViewShape().GetDim(0)),
             return false);
    OP_CHECK(x->GetViewShape().GetDim(0) == indices->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim[0](%ld) and indices' dim[0](%ld) should be the same.",
                     x->GetViewShape().GetDim(0), indices->GetViewShape().GetDim(0)),
             return false);
    OP_CHECK(xGrad->GetViewShape().GetDimNum() == EXPECTED_X_DIM_NUM &&
                 xGrad->GetViewShape().GetDim(0) == x->GetViewShape().GetDim(0) &&
                 xGrad->GetViewShape().GetDim(1) == x->GetViewShape().GetDim(1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad's shape should be the same as x's shape."), return false);
    OP_CHECK(sGrad->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM &&
                 sGrad->GetViewShape().GetDim(0) == s->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "s_grad's shape should be the same as s's shape."), return false);
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* yGrad, const aclTensor* x, const aclTensor* s,
                                      const aclTensor* indices, const aclTensor* xGrad, const aclTensor* sGrad)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(yGrad, x, s, indices, xGrad, sGrad), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(yGrad, x, s, indices, xGrad, sGrad), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape约束
    CHECK_RET(CheckShape(yGrad, x, s, indices, xGrad, sGrad), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuScatterAddBwdGetWorkspaceSize(const aclTensor* yGrad, const aclTensor* x, const aclTensor* s,
                                                  const aclTensor* indices, const aclTensor* xGrad,
                                                  const aclTensor* sGrad, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // 固定写法，参数检查
    L2_DFX_PHASE_1(aclnnNpuScatterAddBwd, DFX_IN(yGrad, x, s, indices), DFX_OUT(xGrad, sGrad));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
             return ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(yGrad, x, s, indices, xGrad, sGrad);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        // 无有效行时为no-op
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto yGradContiguous = l0op::Contiguous(yGrad, uniqueExecutor.get());
    CHECK_RET(yGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sContiguous = l0op::Contiguous(s, uniqueExecutor.get());
    CHECK_RET(sContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xGradContiguous = l0op::Contiguous(xGrad, uniqueExecutor.get());
    CHECK_RET(xGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sGradContiguous = l0op::Contiguous(sGrad, uniqueExecutor.get());
    CHECK_RET(sGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行NpuScatterAddBwd计算，结果写入x_grad、s_grad
    auto scatterBwdOut = l0op::NpuScatterAddBwdAiCore(yGradContiguous, xContiguous, sContiguous, indicesContiguous,
                                                      xGradContiguous, sGradContiguous, uniqueExecutor.get());
    auto xGradResult = std::get<0>(scatterBwdOut);
    auto sGradResult = std::get<1>(scatterBwdOut);
    CHECK_RET(xGradResult != nullptr && sGradResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出非连续时，结果累积在连续副本上，需拷回原始输出
    if (xGradContiguous != xGrad) {
        auto viewCopyResult = l0op::ViewCopy(xGradResult, xGrad, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (sGradContiguous != sGrad) {
        auto viewCopyResult = l0op::ViewCopy(sGradResult, sGrad, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把uniqueExecutor持有的executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuScatterAddBwd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnNpuScatterAddBwd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
