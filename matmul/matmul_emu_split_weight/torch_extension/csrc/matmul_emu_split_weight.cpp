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
#include "aclnnop/aclnn_matmul_emu_split_weight.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace matmul {

static constexpr int64_t DIM_NUM = 2;
static constexpr int8_t Y_DTYPE_FP32 = 0;
static constexpr float EXPECTED_W_LOW_SCALE = 0.00390625f;

at::Tensor matmul_emu_split_weight(const at::Tensor& x, const at::Tensor& w_high, const at::Tensor& w_low,
                                   double w_low_scale, int64_t y_dtype)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "The input x must be on an NPU device, but got ", x.device());
    TORCH_CHECK(w_high.device().type() == at::kPrivateUse1, "The input w_high must be on an NPU device, but got ",
                w_high.device());
    TORCH_CHECK(w_low.device().type() == at::kPrivateUse1, "The input w_low must be on an NPU device, but got ",
                w_low.device());
    TORCH_CHECK(x.device() == w_high.device() && x.device() == w_low.device(),
                "All inputs must be on the same device, but got x=", x.device(), " w_high=", w_high.device(),
                " w_low=", w_low.device());
    TORCH_CHECK(x.dim() == DIM_NUM, "The input x should be 2D, but got ", x.dim(), "D");
    TORCH_CHECK(w_high.dim() == DIM_NUM, "The input w_high should be 2D, but got ", w_high.dim(), "D");
    TORCH_CHECK(w_low.dim() == DIM_NUM, "The input w_low should be 2D, but got ", w_low.dim(), "D");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "The input x must be bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(w_high.scalar_type() == at::kBFloat16, "The input w_high must be bfloat16, but got ",
                w_high.scalar_type());
    TORCH_CHECK(w_low.scalar_type() == at::kBFloat16, "The input w_low must be bfloat16, but got ",
                w_low.scalar_type());
    TORCH_CHECK(w_high.sizes() == w_low.sizes(), "w_low shape must match w_high shape, but got ", w_low.sizes(), " vs ",
                w_high.sizes());
    TORCH_CHECK(y_dtype == Y_DTYPE_FP32, "y_dtype only supports 0(FP32), but got ", y_dtype);
    TORCH_CHECK(std::abs(static_cast<float>(w_low_scale) - EXPECTED_W_LOW_SCALE) < 1e-7f,
                "w_low_scale only supports 1/256 (0.00390625), but got ", w_low_scale);

    at::ScalarType out_dtype = at::kFloat;
    at::Tensor y = at::empty({x.size(0), w_high.size(1)}, x.options().dtype(out_dtype));

    float w_low_scale_f = static_cast<float>(w_low_scale);
    int8_t y_dtype_i8 = static_cast<int8_t>(y_dtype);
    ACLNN_CMD(aclnnMatmulEmuSplitWeight, x, w_high, w_low, y, w_low_scale_f, y_dtype_i8);

    return y;
}

} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_emu_split_weight", &cann_ops_nn::matmul::matmul_emu_split_weight,
          "MatmulEmuSplitWeight operator on NPU");
}
