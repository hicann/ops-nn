/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

// dst_type: 36=FP8_E4M3FN, 35=FP8_E5M2 (matching CANN aclDataType values)
constexpr int64_t DST_TYPE_E4M3FN = 36;
constexpr int64_t DST_TYPE_E5M2 = 35;

std::tuple<at::Tensor, at::Tensor> situ_mx_quant(const at::Tensor& x, double beta, double linearBeta, bool activateLeft,
                                                 int64_t dstType, const std::string& roundMode)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1-dimensional, but got ", x.dim());
    TORCH_CHECK(x.size(-1) % 2 == 0, "x last dim must be even, but got ", x.size(-1));
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(beta > 0.0, "beta must be greater than 0, but got ", beta);
    TORCH_CHECK(dstType == DST_TYPE_E4M3FN || dstType == DST_TYPE_E5M2,
                "dst_type must be 36(E4M3FN) or 35(E5M2), but got ", dstType);
    TORCH_CHECK(roundMode == "rint", "round_mode must be 'rint' for FP8 output, but got ", roundMode);

    // Output y: same shape as x but last dim halved
    std::vector<int64_t> yShape(x.sizes().begin(), x.sizes().end());
    yShape[x.dim() - 1] = x.size(x.dim() - 1) / 2;

    // Output y_scale: y shape with axis dim = CeilDiv(yDim, 64), append dim 2
    constexpr int64_t blockSize = 32;
    constexpr int64_t alignNum = 2;
    int64_t yAxisSize = (yShape[x.dim() - 1] + alignNum * blockSize - 1) / (alignNum * blockSize);
    std::vector<int64_t> yScaleShape(yShape.begin(), yShape.end());
    yScaleShape[x.dim() - 1] = yAxisSize;
    yScaleShape.push_back(alignNum);

    // Determine torch dtypes for output tensors
    at::ScalarType yScalarType = (dstType == DST_TYPE_E5M2) ? at::ScalarType::Float8_e5m2 :
                                                              at::ScalarType::Float8_e4m3fn;
    at::ScalarType yScaleScalarType = at::ScalarType::Float8_e8m0fnu;

    at::Tensor y = at::empty(yShape, x.options().dtype(yScalarType));
    at::Tensor yScale = at::empty(yScaleShape, x.options().dtype(yScaleScalarType));

    int64_t axis = -1;

    ACLNN_CMD(aclnnSituMxQuant, x, beta, linearBeta, activateLeft, axis, dstType, roundMode, y, yScale);

    return std::make_tuple(y, yScale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_mx_quant", &cann_ops_nn::quant::situ_mx_quant, "SituMxQuant on NPU");
}
