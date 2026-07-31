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
#include "aclnnop/aclnn_rms_norm_dynamic_quant.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace norm {

std::tuple<at::Tensor, at::Tensor> rms_norm_dynamic_quant(const at::Tensor& x, const at::Tensor& gamma,
                                                          const c10::optional<at::Tensor>& smooth_scales,
                                                          const c10::optional<at::Tensor>& beta, double epsilon,
                                                          int64_t dst_type)
{
    // 1. 设备检查
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device");
    TORCH_CHECK(gamma.device().type() == at::kPrivateUse1, "gamma must be on NPU device");

    // 2. 维度检查
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 8, "x must be 2-8 dimensional, but got ", x.dim());
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1-dimensional, but got ", gamma.dim());
    TORCH_CHECK(gamma.size(0) == x.size(-1), "gamma size (", gamma.size(0), ") must match x last dimension (",
                x.size(-1), ")");

    // 3. 数据类型检查
    auto input_dtype = x.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(gamma.scalar_type() == input_dtype, "gamma dtype must match x dtype: ", input_dtype, " vs ",
                gamma.scalar_type());

    if (smooth_scales.has_value() && smooth_scales.value().defined()) {
        TORCH_CHECK(smooth_scales.value().device().type() == at::kPrivateUse1, "smooth_scales must be on NPU");
        TORCH_CHECK(smooth_scales.value().scalar_type() == input_dtype, "smooth_scales dtype mismatch");
        TORCH_CHECK(smooth_scales.value().dim() == 1, "smooth_scales must be 1D");
        TORCH_CHECK(smooth_scales.value().size(0) == x.size(-1), "smooth_scales size mismatch");
    }

    if (beta.has_value() && beta.value().defined()) {
        TORCH_CHECK(beta.value().device().type() == at::kPrivateUse1, "beta must be on NPU");
        TORCH_CHECK(beta.value().scalar_type() == input_dtype, "beta dtype mismatch");
        TORCH_CHECK(beta.value().dim() == 1, "beta must be 1D");
        TORCH_CHECK(beta.value().size(0) == x.size(-1), "beta size mismatch");
    }

    // 4. 输出形状
    std::vector<int64_t> y_shape(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> scale_shape(x.sizes().begin(), x.sizes().end() - 1);

    // 5. 创建输出张量
    at::Tensor y = at::empty(y_shape, at::TensorOptions().dtype(at::kChar).device(at::kPrivateUse1));
    at::Tensor scale = at::empty(scale_shape, at::TensorOptions().dtype(at::kFloat).device(at::kPrivateUse1));

    // 6. ACLNN 调用
    ACLNN_CMD(aclnnRmsNormDynamicQuant, x, gamma, smooth_scales, beta, epsilon, dst_type, y, scale);

    return std::make_tuple(y, scale);
}

} // namespace norm
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rms_norm_dynamic_quant", &cann_ops_nn::norm::rms_norm_dynamic_quant,
          "RmsNormDynamicQuant INT8 quantization on NPU");
}
