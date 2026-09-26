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
 * \file aclnn_npu_scatter_add.cpp
 * \brief aclnnNpuScatterAdd 两段式接口实现
 */
#include "aclnn_npu_scatter_add.h"
#include "npu_scatter_add.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t EXPECTED_X_DIM_NUM = 2;
static constexpr size_t EXPECTED_Y_DIM_NUM = 2;
static constexpr size_t EXPECTED_INDEX_DIM_NUM = 1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_BF16, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* y, const aclTensor* indices,
                                const aclTensor* sortIdx)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(sortIdx, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* x, const aclTensor* y, const aclTensor* s, const aclTensor* indices,
                                   const aclTensor* sortIdx, const aclTensor* validTokenNum)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    if (s != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(s, x->GetDataType(), return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(sortIdx, INDEX_DTYPE_SUPPORT_LIST, return false);
    if (validTokenNum != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(validTokenNum, INDEX_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y, const aclTensor* s, const aclTensor* indices,
                       const aclTensor* sortIdx, const aclTensor* validTokenNum)
{
    OP_CHECK(
        x->GetViewShape().GetDimNum() == EXPECTED_X_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x should be a 2d tensor, but got %zu dims.", x->GetViewShape().GetDimNum()),
        return false);
    OP_CHECK(
        y->GetViewShape().GetDimNum() == EXPECTED_Y_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y should be a 2d tensor, but got %zu dims.", y->GetViewShape().GetDimNum()),
        return false);
    OP_CHECK(indices->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices should be a 1d tensor, but got %zu dims.",
                     indices->GetViewShape().GetDimNum()),
             return false);
    OP_CHECK(sortIdx->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sort_idx should be a 1d tensor, but got %zu dims.",
                     sortIdx->GetViewShape().GetDimNum()),
             return false);

    OP_CHECK(x->GetViewShape().GetDim(1) == y->GetViewShape().GetDim(1),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim[1](%ld) and y's dim[1](%ld) should be the same.",
                     x->GetViewShape().GetDim(1), y->GetViewShape().GetDim(1)),
             return false);
    OP_CHECK(x->GetViewShape().GetDim(0) == indices->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim[0](%ld) and indices' dim[0](%ld) should be the same.",
                     x->GetViewShape().GetDim(0), indices->GetViewShape().GetDim(0)),
             return false);
    OP_CHECK(x->GetViewShape().GetDim(0) == sortIdx->GetViewShape().GetDim(0),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x's dim[0](%ld) and sort_idx's dim[0](%ld) should be the same.",
                     x->GetViewShape().GetDim(0), sortIdx->GetViewShape().GetDim(0)),
             return false);

    if (s != nullptr) {
        OP_CHECK(s->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "s should be a 1d tensor, but got %zu dims.",
                         s->GetViewShape().GetDimNum()),
                 return false);
        OP_CHECK(s->GetViewShape().GetDim(0) == x->GetViewShape().GetDim(0),
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "s's dim[0](%ld) and x's dim[0](%ld) should be the same.",
                         s->GetViewShape().GetDim(0), x->GetViewShape().GetDim(0)),
                 return false);
    }

    if (validTokenNum != nullptr) {
        OP_CHECK(validTokenNum->GetViewShape().GetDimNum() == EXPECTED_INDEX_DIM_NUM &&
                     validTokenNum->GetViewShape().GetDim(0) == 1,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "valid_token_num should be a 1d tensor with 1 element."),
                 return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor* x, const aclTensor* y, const aclTensor* s,
                                      const aclTensor* indices, const aclTensor* sortIdx,
                                      const aclTensor* validTokenNum)
{
    // 1. 检查参数是否为空指针（s、valid_token_num为可选输入，允许为空）
    CHECK_RET(CheckNotNull(x, y, indices, sortIdx), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, y, s, indices, sortIdx, validTokenNum), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape约束
    CHECK_RET(CheckShape(x, y, s, indices, sortIdx, validTokenNum), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuScatterAddGetWorkspaceSize(const aclTensor* x, const aclTensor* y, const aclTensor* s,
                                               const aclTensor* indices, const aclTensor* sortIdx,
                                               const aclTensor* validTokenNum, bool useHighPrecision,
                                               uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    // 固定写法，参数检查
    L2_DFX_PHASE_1(aclnnNpuScatterAdd, DFX_IN(x, y, s, indices, sortIdx, validTokenNum), DFX_OUT(y));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    OP_CHECK(uniqueExecutor.get() != nullptr, OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "Create executor error."),
             return ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, y, s, indices, sortIdx, validTokenNum);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        // 无有效行时为no-op，y保持不变
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入转换成连续的tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto yContiguous = l0op::Contiguous(y, uniqueExecutor.get());
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sortIdxContiguous = l0op::Contiguous(sortIdx, uniqueExecutor.get());
    CHECK_RET(sortIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* sContiguous = s;
    if (s != nullptr) {
        sContiguous = l0op::Contiguous(s, uniqueExecutor.get());
        CHECK_RET(sContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    const aclTensor* validTokenNumContiguous = validTokenNum;
    if (validTokenNum != nullptr) {
        validTokenNumContiguous = l0op::Contiguous(validTokenNum, uniqueExecutor.get());
        CHECK_RET(validTokenNumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 进行NpuScatterAdd计算，结果inplace写入yContiguous
    auto scatterOut = l0op::NpuScatterAddAiCore(xContiguous, yContiguous, sContiguous, indicesContiguous,
                                                sortIdxContiguous, validTokenNumContiguous, useHighPrecision,
                                                uniqueExecutor.get());
    CHECK_RET(scatterOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // y非连续时，结果累积在连续副本上，需拷回原始y
    if (yContiguous != y) {
        auto viewCopyResult = l0op::ViewCopy(scatterOut, y, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把uniqueExecutor持有的executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNpuScatterAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnNpuScatterAdd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
