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

#include "aclnn_data_format_dim_map.h"
#include "data_format_dim_map.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_INT64
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(y, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_MATCH(y, x->GetDataType(), return false);
    OP_CHECK(IsDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: INT32, INT64.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* y)
{
    auto formatX = x->GetStorageFormat();
    auto formatY = y->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(formatX) || IsPrivateFormat(formatY)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: x=%d, y=%d",
                     static_cast<int>(formatX), static_cast<int>(formatY)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(y, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const char* srcFormat,
                                const char* dstFormat, const aclTensor* y)
{
    CHECK_COND(CheckNotNull(x, y), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(srcFormat != nullptr, ACLNN_ERR_PARAM_NULLPTR, "srcFormat is null");
    CHECK_COND(dstFormat != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dstFormat is null");
    CHECK_COND(CheckDtypeValid(x, y), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x_dtype=%d, y_dtype=%d",
               static_cast<int>(x->GetDataType()), static_cast<int>(y->GetDataType()));
    CHECK_COND(CheckFormat(x, y), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
    CHECK_COND(CheckShape(x, y), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnDataFormatDimMapGetWorkspaceSize(
    const aclTensor* x,
    const char* srcFormat,
    const char* dstFormat,
    const aclTensor* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDataFormatDimMap, DFX_IN(x, srcFormat, dstFormat), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, srcFormat, dstFormat, y);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::DataFormatDimMap(
        xContiguous, srcFormat, dstFormat, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnDataFormatDimMap(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDataFormatDimMap);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
