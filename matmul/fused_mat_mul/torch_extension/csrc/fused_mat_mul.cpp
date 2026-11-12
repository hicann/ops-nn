/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {
namespace {
constexpr int64_t X_DIM = 2;
constexpr size_t FUSED_TYPE_ARRAY_SIZE = 100;
constexpr int8_t USE_FP32_ADD = 4;

aclScalar* CreateOptionalFloatScalar(const c10::optional<double>& value)
{
    if (!value.has_value()) {
        return nullptr;
    }
    static const auto aclCreateScalar = GET_OP_API_FUNC(CreateScalar);
    TORCH_CHECK(aclCreateScalar != nullptr, "aclCreateScalar is not available");
    float scalarValue = static_cast<float>(value.value());
    return aclCreateScalar(&scalarValue, ACL_FLOAT);
}

bool NeedFusedAddMulHighPrecision(const at::Tensor& x, const at::Tensor& x2, const c10::optional<at::Tensor>& x3,
                                  c10::string_view fusedOpType)
{
    static const char* compatibleEnv = std::getenv("TORCH_NPU_USE_COMPATIBLE_IMPL");
    const bool compatibleImplEnabled = compatibleEnv != nullptr && std::string(compatibleEnv) == "1";
    const bool isAddOrMul = fusedOpType == c10::string_view("add") || fusedOpType == c10::string_view("mul");
    const bool isLowPrecision = x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16;
    if (!compatibleImplEnabled || !isAddOrMul || !isLowPrecision || x.scalar_type() != x2.scalar_type() ||
        !x3.has_value() || !x3->defined()) {
        return false;
    }
    return x3->scalar_type() == x.scalar_type();
}

std::vector<int64_t> InferOutputShape(const at::Tensor& x, const at::Tensor& x2)
{
    TORCH_CHECK(x.dim() >= X_DIM, "x must have at least ", X_DIM, " dimensions, but got ", x.dim());
    TORCH_CHECK(x2.dim() >= X_DIM, "x2 must have at least ", X_DIM, " dimensions, but got ", x2.dim());
    TORCH_CHECK(x.size(x.dim() - 1) == x2.size(x2.dim() - X_DIM), "x and x2 K dimensions must match");

    std::vector<int64_t> outputShape;
    const int64_t xBatchDims = x.dim() - X_DIM;
    const int64_t x2BatchDims = x2.dim() - X_DIM;
    const int64_t outputBatchDims = std::max(xBatchDims, x2BatchDims);
    outputShape.reserve(outputBatchDims + X_DIM);
    for (int64_t i = 0; i < outputBatchDims; ++i) {
        const int64_t xIndex = i - (outputBatchDims - xBatchDims);
        const int64_t x2Index = i - (outputBatchDims - x2BatchDims);
        const int64_t xDim = xIndex < 0 ? 1 : x.size(xIndex);
        const int64_t x2Dim = x2Index < 0 ? 1 : x2.size(x2Index);
        TORCH_CHECK(xDim == x2Dim || xDim == 1 || x2Dim == 1, "x and x2 batch dimensions are not broadcastable");
        outputShape.push_back(std::max(xDim, x2Dim));
    }
    outputShape.push_back(x.size(x.dim() - X_DIM));
    outputShape.push_back(x2.size(x2.dim() - 1));
    return outputShape;
}

void CheckFusedMatmulInputs(const at::Tensor& x, const at::Tensor& x2, const c10::optional<at::Tensor>& x3,
                            c10::string_view fusedOpType)
{
    TORCH_CHECK(x.scalar_type() == x2.scalar_type(), "x and x2 must have the same dtype");
    const bool hasX3 = x3.has_value() && x3->defined();
    const bool supportsX3 = fusedOpType == c10::string_view("add") || fusedOpType == c10::string_view("mul");
    if (supportsX3) {
        TORCH_CHECK(hasX3, "x3 must be provided for add or mul");
    } else {
        TORCH_CHECK(!hasX3, "x3 is only supported for add or mul");
    }
}

} // namespace

at::Tensor fused_matmul(const at::Tensor& x, const at::Tensor& x2, const c10::optional<at::Tensor>& bias,
                        const c10::optional<at::Tensor>& x3, const c10::optional<double>& alpha,
                        const c10::optional<double>& beta, c10::string_view fusedOpType, int64_t cubeMathType)
{
    TORCH_CHECK(fusedOpType.size() <= FUSED_TYPE_ARRAY_SIZE, "the length of fused_op_type cannot be greater than ",
                FUSED_TYPE_ARRAY_SIZE);
    CheckFusedMatmulInputs(x, x2, x3, fusedOpType);

    auto outputOptions = fusedOpType == c10::string_view("16cast32") ? x.options().dtype(at::kFloat) : x.options();
    auto outputShape = InferOutputShape(x, x2);
    at::Tensor y = at_npu::native::OpPreparation::apply_tensor_without_format(outputShape, outputOptions);
    std::string fusedOpTypeString(fusedOpType.data(), fusedOpType.size());
    int8_t aclCubeMathType = static_cast<int8_t>(cubeMathType);
    if (NeedFusedAddMulHighPrecision(x, x2, x3, fusedOpType)) {
        aclCubeMathType = USE_FP32_ADD;
    }
    auto alphaScalar = CreateOptionalFloatScalar(alpha);
    auto betaScalar = CreateOptionalFloatScalar(beta);
    ACLNN_CMD(aclnnFusedMatmulV2, x, x2, bias, x3, alphaScalar, betaScalar, fusedOpTypeString, aclCubeMathType, y);
    return y;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_matmul", &cann_ops_nn::matmul::fused_matmul, "FusedMatmul on NPU");
}
