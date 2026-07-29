/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_matmul_gelu.h"

#include "fused_matmul_gelu.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MIN_X_DIM_NUM = 2;
constexpr size_t WEIGHT_DIM_NUM = 2;
constexpr size_t BIAS_DIM_NUM = 1;
constexpr size_t LAST_DIM_OFFSET = 1;
constexpr int64_t APPROXIMATE_TANH = 1;

static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};

struct FusedMatmulGeluInputTensor {
    const aclTensor* x;
    const aclTensor* weight;
    const aclTensor* bias;
};

struct FusedMatmulGeluOutputTensor {
    aclTensor* y;
};

inline static bool ShapeEqual(const op::Shape& lhs, const op::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }

    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }

    return true;
}

inline static op::Shape InferYShape(const aclTensor* x, const aclTensor* weight)
{
    op::Shape yShape;
    const auto xShape = x->GetViewShape();
    const auto weightShape = weight->GetViewShape();
    const size_t xDimNum = xShape.GetDimNum();

    for (size_t i = 0; i + LAST_DIM_OFFSET < xDimNum; ++i) {
        yShape.AppendDim(xShape.GetDim(i));
    }
    yShape.AppendDim(weightShape.GetDim(0));

    return yShape;
}

inline static int64_t GetLastDim(const op::Shape& shape) { return shape.GetDim(shape.GetDimNum() - LAST_DIM_OFFSET); }

inline static bool CheckNotNull(const FusedMatmulGeluInputTensor& inputTensors,
                                const FusedMatmulGeluOutputTensor& outputTensors, const uint64_t* workspaceSize,
                                aclOpExecutor** executor)
{
    OP_CHECK_NULL(inputTensors.x, return false);
    OP_CHECK_NULL(inputTensors.weight, return false);
    OP_CHECK_NULL(outputTensors.y, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

inline static bool CheckDtypeValid(const FusedMatmulGeluInputTensor& inputTensors,
                                   const FusedMatmulGeluOutputTensor& outputTensors)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.x, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.weight, FLOAT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensors.x, inputTensors.weight, return false);

    if (inputTensors.bias != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensors.bias, FLOAT_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SAME(inputTensors.x, inputTensors.bias, return false);
    }

    OP_CHECK_DTYPE_NOT_SAME(outputTensors.y, inputTensors.x, return false);
    return true;
}

inline static bool CheckShapeValid(const FusedMatmulGeluInputTensor& inputTensors,
                                   const FusedMatmulGeluOutputTensor& outputTensors)
{
    const auto xShape = inputTensors.x->GetViewShape();
    const auto weightShape = inputTensors.weight->GetViewShape();

    if (xShape.GetDimNum() < MIN_X_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x dim number should be greater than or equal to 2, but got %zu.",
                xShape.GetDimNum());
        return false;
    }

    if (weightShape.GetDimNum() != WEIGHT_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight should be 2-D [N, K], but got dim number %zu.",
                weightShape.GetDimNum());
        return false;
    }

    const int64_t kFromX = GetLastDim(xShape);
    const int64_t nFromWeight = weightShape.GetDim(0);
    const int64_t kFromWeight = weightShape.GetDim(1);

    if (kFromX != kFromWeight) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x last dim K should equal weight dim1 K, but got x K %ld, weight K %ld.",
                kFromX, kFromWeight);
        return false;
    }

    if (nFromWeight <= 0 || kFromWeight <= 0 || kFromX <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "N/K should be greater than 0, but got N %ld, K %ld.", nFromWeight,
                kFromWeight);
        return false;
    }

    if (inputTensors.bias != nullptr) {
        const auto biasShape = inputTensors.bias->GetViewShape();
        if (biasShape.GetDimNum() != BIAS_DIM_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias should be 1-D [N], but got dim number %zu.", biasShape.GetDimNum());
            return false;
        }
        if (biasShape.GetDim(0) != nFromWeight) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias shape should be [N], but got bias N %ld, weight N %ld.",
                    biasShape.GetDim(0), nFromWeight);
            return false;
        }
    }

    const auto expectedYShape = InferYShape(inputTensors.x, inputTensors.weight);
    const auto yShape = outputTensors.y->GetViewShape();
    if (!ShapeEqual(yShape, expectedYShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "y shape is invalid, expected %s, but got %s.",
                op::ToString(expectedYShape).GetString(), op::ToString(yShape).GetString());
        return false;
    }

    return true;
}

inline static bool CheckAttrValid(int64_t approximate)
{
    if (approximate != APPROXIMATE_TANH) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "approximate should be 1(tanh), but got %ld.", approximate);
        return false;
    }
    return true;
}

inline static aclnnStatus CheckParam(const FusedMatmulGeluInputTensor& inputTensors,
                                     const FusedMatmulGeluOutputTensor& outputTensors, int64_t approximate,
                                     const uint64_t* workspaceSize, aclOpExecutor** executor)
{
    CHECK_RET(CheckNotNull(inputTensors, outputTensors, workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(inputTensors, outputTensors), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(inputTensors, outputTensors), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttrValid(approximate), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnFusedMatmulGeluGetWorkspaceSize(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                                 int64_t approximate, aclTensor* y, uint64_t* workspaceSize,
                                                 aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnFusedMatmulGelu, DFX_IN(x, weight, bias, approximate), DFX_OUT(y));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    FusedMatmulGeluInputTensor inputTensors = {x, weight, bias};
    FusedMatmulGeluOutputTensor outputTensors = {y};

    auto ret = CheckParam(inputTensors, outputTensors, approximate, workspaceSize, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* biasContiguous = nullptr;
    if (bias != nullptr) {
        biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
        CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto fusedOut = l0op::FusedMatmulGelu(xContiguous, weightContiguous, biasContiguous, approximate,
                                          uniqueExecutor.get());
    CHECK_RET(fusedOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fusedOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFusedMatmulGelu(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFusedMatmulGelu);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
