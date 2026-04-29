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
 * \file aclnn_threshold_v2.cpp
 * \brief ACLNN L2 API for ThresholdV2
 */
#include "aclnn_threshold_v2.h"
#include "threshold_v2.h"
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
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* self, const aclScalar* threshold,
                          const aclScalar* value, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(threshold, return false);
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK(IsDtypeSupported(self->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "self dtype not supported: dtype=%d. Supports FLOAT/FLOAT16/BFLOAT16.",
                     static_cast<int>(self->GetDataType())),
             return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(self, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);

    auto selfShape = self->GetViewShape();
    auto outShape = out->GetViewShape();
    OP_CHECK(selfShape == outShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "self and out must have the same shape."),
             return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclScalar* threshold,
                                const aclScalar* value, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(self, threshold, value, out),
               ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: self_dtype=%d, out_dtype=%d",
               static_cast<int>(self->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: self_dim=%zu, out_dim=%zu",
               self->GetViewShape().GetDimNum(), out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnThresholdV2GetWorkspaceSize(
    const aclTensor* self,
    const aclScalar* threshold,
    const aclScalar* value,
    aclTensor*       out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
{
    L2_DFX_PHASE_1(aclnnThresholdV2, DFX_IN(self, threshold, value), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, threshold, value, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    float thresholdF = threshold->ToFloat();
    float valueF = value->ToFloat();

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::ThresholdV2(selfContiguous, thresholdF, valueF,
                                                    uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnThresholdV2(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnThresholdV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
