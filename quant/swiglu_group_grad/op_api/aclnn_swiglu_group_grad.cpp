/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_swiglu_group_grad.h"
#include "swiglu_group_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include <string>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr const char* ACLNN_SWIGLU_GROUP_GRAD_NAME = "aclnnSwigluGroupGrad";
constexpr int64_t SPLIT_NUM = 2;

// ── Supported dtype list ───────────────────────────────────────────────────
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT,
                                                                   DataType::DT_BF16};

// ── Null-pointer checks ────────────────────────────────────────────────────
static inline bool CheckNotNull(const aclTensor* gradY, const aclTensor* x, const aclTensor* gradXOut,
                                const aclTensor* gradWeightOutOptional, const aclTensor* weightOptional)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gradXOut, return false);

    // weightOptional non-null ⇒ gradWeightOutOptional must be non-null
    if (weightOptional != nullptr) {
        OP_CHECK_NULL(gradWeightOutOptional, return false);
    }
    return true;
}

// ── Dtype checks ───────────────────────────────────────────────────────────
static inline bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* x, const aclTensor* weightOptional,
                                   const aclTensor* yOriginOptional, const aclTensor* groupIndexOptional,
                                   const aclTensor* gradXOut, const aclTensor* gradWeightOutOptional)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradY, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradY, x, return false);
    OP_CHECK_DTYPE_NOT_SAME(gradY, gradXOut, return false);

    if (weightOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(weightOptional, DataType::DT_FLOAT, return false);
    }
    if (yOriginOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(gradY, yOriginOptional, return false);
    }
    if (groupIndexOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(groupIndexOptional, DataType::DT_INT64, return false);
    }
    if (gradWeightOutOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(gradWeightOutOptional, DataType::DT_FLOAT, return false);
    }
    return true;
}

// ── Shape checks ───────────────────────────────────────────────────────────
static inline bool CheckShape(const aclTensor* gradY, const aclTensor* x, const aclTensor* weightOptional,
                              const aclTensor* yOriginOptional, const aclTensor* groupIndexOptional,
                              const aclTensor* gradXOut, const aclTensor* gradWeightOutOptional)
{
    auto gradYShape = gradY->GetViewShape();
    const size_t inputRank = gradYShape.GetDimNum();
    if (inputRank < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY must be at least 1D, got %zu dims.", inputRank);
        return false;
    }
    const int64_t H = gradYShape.GetDim(inputRank - 1);
    int64_t totalRows = 1;
    for (size_t i = 0; i < inputRank - 1; ++i) {
        totalRows *= gradYShape.GetDim(i);
    }

    auto xShape = x->GetViewShape();
    if (xShape.GetDimNum() < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x must be at least 1D, got %zu dims.", xShape.GetDimNum());
        return false;
    }
    if (xShape.GetDim(xShape.GetDimNum() - 1) != H * SPLIT_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x.shape[-1]=%ld != 2*H=%ld", xShape.GetDim(xShape.GetDimNum() - 1),
                H * SPLIT_NUM);
        return false;
    }
    int64_t xTotalRows = 1;
    for (size_t i = 0; i < xShape.GetDimNum() - 1; ++i) {
        xTotalRows *= xShape.GetDim(i);
    }
    if (xTotalRows != totalRows) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x outer numel(%ld) must equal gradY outer numel(%ld).", xTotalRows,
                totalRows);
        return false;
    }

    if (H <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "H=%ld must be > 0", H);
        return false;
    }

    OP_CHECK_SHAPE_NOT_EQUAL(gradXOut, x, return false);

    if (weightOptional != nullptr) {
        auto weightShape = weightOptional->GetViewShape();
        int64_t weightElementNum = weightShape.GetShapeSize();
        if (weightElementNum != totalRows) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                ACLNN_SWIGLU_GROUP_GRAD_NAME, "weightOptional", std::to_string(weightElementNum).c_str(),
                "The element num of weightOptional must be equal to the product of gradY leading dims.");
            return false;
        }
        OP_CHECK_SHAPE_NOT_EQUAL(gradWeightOutOptional, weightOptional, return false);
    }

    if (yOriginOptional != nullptr) {
        auto yOriginShape = yOriginOptional->GetViewShape();
        if (yOriginShape.GetDimNum() < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOriginOptional must be at least 1D.");
            return false;
        }
        if (yOriginShape.GetDim(yOriginShape.GetDimNum() - 1) != H) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOriginOptional.shape[-1]=%ld must equal H=%ld.",
                    yOriginShape.GetDim(yOriginShape.GetDimNum() - 1), H);
            return false;
        }
        int64_t yOriginTotalRows = 1;
        for (size_t i = 0; i < yOriginShape.GetDimNum() - 1; ++i) {
            yOriginTotalRows *= yOriginShape.GetDim(i);
        }
        if (yOriginTotalRows != totalRows) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "yOriginOptional outer numel(%ld) must equal gradY outer numel(%ld).",
                    yOriginTotalRows, totalRows);
            return false;
        }
    }

    if (groupIndexOptional != nullptr) {
        auto groupIndexShape = groupIndexOptional->GetViewShape();
        if (groupIndexShape.GetDimNum() != 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupIndexOptional must be 1D, got %zu dims.",
                    groupIndexShape.GetDimNum());
            return false;
        }
        if (groupIndexShape.GetDim(0) < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupIndexOptional must have numel>=1, got %ld.",
                    groupIndexShape.GetDim(0));
            return false;
        }
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradY, const aclTensor* x, const aclTensor* weightOptional,
                               const aclTensor* yOriginOptional, const aclTensor* groupIndexOptional, float clampLimit,
                               const aclTensor* gradXOut, const aclTensor* gradWeightOutOptional)
{
    CHECK_RET(CheckNotNull(gradY, x, gradXOut, gradWeightOutOptional, weightOptional), ACLNN_ERR_PARAM_NULLPTR);
    if ((weightOptional == nullptr) != (yOriginOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weightOptional and yOriginOptional must be provided together.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (clampLimit != -1.0f && !(clampLimit > 0.0f)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "clampLimit must be -1.0 (no clamp) or > 0.0, but got %f.", clampLimit);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(
        CheckDtypeValid(gradY, x, weightOptional, yOriginOptional, groupIndexOptional, gradXOut, gradWeightOutOptional),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShape(gradY, x, weightOptional, yOriginOptional, groupIndexOptional, gradXOut, gradWeightOutOptional),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// ── GetWorkspaceSize implementation ─────────────────────────────────────────
aclnnStatus aclnnSwigluGroupGradGetWorkspaceSize(const aclTensor* gradY, const aclTensor* x,
                                                 const aclTensor* weightOptional, const aclTensor* yOriginOptional,
                                                 const aclTensor* groupIndexOptional, float clampLimit,
                                                 aclTensor* gradXOut, aclTensor* gradWeightOutOptional,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnSwigluGroupGrad, DFX_IN(gradY, x, weightOptional, yOriginOptional, groupIndexOptional),
                   DFX_OUT(gradXOut, gradWeightOutOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradY, x, weightOptional, yOriginOptional, groupIndexOptional, clampLimit, gradXOut,
                           gradWeightOutOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Empty tensor handling
    if (gradY->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // Make inputs contiguous
    auto contiguousGradOutput = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(contiguousGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto contiguousX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(contiguousX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Make optional inputs contiguous when present
    const aclTensor* contiguousWeight = nullptr;
    if (weightOptional != nullptr) {
        contiguousWeight = l0op::Contiguous(weightOptional, uniqueExecutor.get());
        CHECK_RET(contiguousWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* contiguousYOrigin = nullptr;
    if (yOriginOptional != nullptr) {
        contiguousYOrigin = l0op::Contiguous(yOriginOptional, uniqueExecutor.get());
        CHECK_RET(contiguousYOrigin != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* contiguousGroupIndex = nullptr;
    if (groupIndexOptional != nullptr) {
        contiguousGroupIndex = l0op::Contiguous(groupIndexOptional, uniqueExecutor.get());
        CHECK_RET(contiguousGroupIndex != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Call kernel via l0op::SwigluGroupGrad
    auto result = l0op::SwigluGroupGrad(contiguousGradOutput, contiguousX, contiguousWeight, contiguousYOrigin,
                                        contiguousGroupIndex, clampLimit, uniqueExecutor.get());
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy gradXOut result to user output tensor
    auto viewCopyGradX = l0op::ViewCopy(result[0], gradXOut, uniqueExecutor.get());
    CHECK_RET(viewCopyGradX != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy gradWeightOutOptional when present
    if (gradWeightOutOptional != nullptr && weightOptional != nullptr) {
        CHECK_RET(result[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyGradWeight = l0op::ViewCopy(result[1], gradWeightOutOptional, uniqueExecutor.get());
        CHECK_RET(viewCopyGradWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// ── Second-stage interface ──────────────────────────────────────────────────
aclnnStatus aclnnSwigluGroupGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSwigluGroupGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
