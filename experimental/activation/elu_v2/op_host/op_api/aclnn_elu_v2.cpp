/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_elu_v2.h"
#include "elu_v2.h"
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

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16,
                                                                              DataType::DT_BF16};

static bool IsDtypeSupported(DataType dtype) { return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST); }

static bool CheckNotNull(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                         const aclScalar* inputScale, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(scale, return false);
    OP_CHECK_NULL(inputScale, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}
static bool CheckDtypeValid(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                            const aclScalar* inputScale, const aclTensor* out)
{
    // Output dtype must match input dtype
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);

    if (!IsDtypeSupported(self->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Dtype not supported: dtype=%d. "
                "Supported: FLOAT16, FLOAT, BF16.",
                static_cast<int>(self->GetDataType()));
        return false;
    }

    // 检查scale的数据类型能否转换为FLOAT
    if (!CanCast(scale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale dtype %s can not cast to float32.",
                ToString(scale->GetDataType()).GetString());
        return false;
    }

    // 检查alpha的数据类型能否转换为FLOAT
    if (!CanCast(alpha->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha dtype %s can not cast to float32.",
                ToString(alpha->GetDataType()).GetString());
        return false;
    }

    // 检查inputScale的数据类型能否转换为FLOAT
    if (!CanCast(inputScale->GetDataType(), DataType::DT_FLOAT)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inputScale dtype %s can not cast to float32.",
                ToString(inputScale->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensor* out)
{
    auto formatSelf = self->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();

    if (IsPrivateFormat(formatSelf) || IsPrivateFormat(formatOut)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Private format not supported: self=%d, out=%d", static_cast<int>(formatSelf),
                static_cast<int>(formatOut));
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(out, self, return false);
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                               const aclScalar* inputScale, const aclTensor* out)
{
    if (!CheckNotNull(self, alpha, scale, inputScale, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!CheckDtypeValid(self, alpha, scale, inputScale, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
                static_cast<int>(self->GetDataType()), static_cast<int>(out->GetDataType()));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckFormat(self, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckShape(self, out)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckShape failed: self_dim=%zu, out_dim=%zu",
                self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnEluV2GetWorkspaceSize(const aclTensor* self, const aclScalar* alpha, const aclScalar* scale,
                                                  const aclScalar* inputScale, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnEluV2, DFX_IN(self, alpha, scale, inputScale), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, alpha, scale, inputScale, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果self是空tensor，则out也是空tensor，直接返回
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    auto contiguousSelf = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(contiguousSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::EluV2(contiguousSelf, alpha->ToFloat(), scale->ToFloat(), inputScale->ToFloat(),
                                            uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnEluV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEluV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
