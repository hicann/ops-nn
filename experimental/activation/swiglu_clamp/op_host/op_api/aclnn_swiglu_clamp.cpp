/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_swiglu_clamp.cpp
 * \brief SwigluClamp aclnn two-phase API. Framework mirrors aclnnMish; scalar limit and
 *        halved-output shape handling mirror aclnnLeakyRelu / aclnnGlu respectively.
 */
#include "aclnn_swiglu_clamp.h"
#include "swiglu_clamp.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "op_api/op_api_def.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/level2_base.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                              DataType::DT_BF16};

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    auto supportList = GetDtypeSupportListV2(ASCEND910B_DTYPE_SUPPORT_LIST, ASCEND910_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, supportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, supportList, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, y, return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(y, MAX_SUPPORT_DIMS_NUMS, return false);
    // y shape = x shape with the last dim halved
    auto xShape = x->GetViewShape();
    auto yShape = y->GetViewShape();
    if (xShape.GetDimNum() != yShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x/y dim num mismatch: %zu vs %zu", xShape.GetDimNum(), yShape.GetDimNum());
        return false;
    }
    if (xShape.GetDimNum() == 0) {
        return true; // scalar input, nothing more to check
    }
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        int64_t xd = xShape.GetDim(i);
        int64_t yd = yShape.GetDim(i);
        if (i == xShape.GetDimNum() - 1) {
            if (xd != yd * 2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x last dim must be 2*y last dim, got %ld vs %ld", xd, yd);
                return false;
            }
        } else if (xd != yd) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x/y dim %zu mismatch: %ld vs %ld", i, xd, yd);
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const double limit, const aclTensor* y)
{
    // 1. null check (limit is a by-value scalar, no null check needed)
    CHECK_RET(CheckNotNull2Tensor(x, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. dtype in support list and x/y same dtype
    CHECK_RET(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID);

    // 3. shape: y last dim = x last dim / 2
    CHECK_RET(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID);

    // 4. limit must be positive
    OP_CHECK(limit > 0.0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "limit must be > 0, got %f", limit),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSwigluClampGetWorkspaceSize(const aclTensor* x, double limit, aclTensor* y, uint64_t* workspaceSize,
                                             aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSwigluClamp, DFX_IN(x), DFX_OUT(y));

    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // param check
    auto ret = CheckParams(x, limit, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // empty tensor
    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // x may be non-contiguous -> contiguous first
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // SwigluClamp kernel (limit cast to float for the l0op signature)
    auto result = l0op::SwigluClamp(xContiguous, static_cast<float>(limit), uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // result may be contiguous; ViewCopy handles a non-contiguous out tensor y
    auto viewCopyResult = l0op::ViewCopy(result, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSwigluClamp(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSwigluClamp);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
