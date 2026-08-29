/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnnop/aclnn_add_rms_norm_dynamic_mx_quant.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace norm {
namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t MXSCALE_TAIL_DIM = 2;
constexpr int64_t DST_TYPE_FP8_E5M2 = 35;
constexpr int64_t DST_TYPE_FP8_E4M3FN = 36;
constexpr int64_t DST_TYPE_FP4_E2M1 = 40;
constexpr int64_t DST_TYPE_FP4_E1M2 = 41;

static aclDataType GetAclDataTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return ACL_FLOAT8_E5M2;
        case DST_TYPE_FP8_E4M3FN:
            return ACL_FLOAT8_E4M3FN;
        case DST_TYPE_FP4_E2M1:
            return ACL_FLOAT4_E2M1;
        case DST_TYPE_FP4_E1M2:
            return ACL_FLOAT4_E1M2;
        default:
            TORCH_CHECK(false,
                        "dst_type must be 35(FP8_E5M2), 36(FP8_E4M3FN), "
                        "40(FP4_E2M1) or 41(FP4_E1M2), but got ",
                        dst_type);
    }
}

static at::ScalarType GetScalarTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case DST_TYPE_FP8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        case DST_TYPE_FP4_E2M1:
        case DST_TYPE_FP4_E1M2:
            return at::kByte; // FP4 packed as uint8
        default:
            TORCH_CHECK(false, "Unsupported dst_type: ", dst_type);
    }
}

static bool IsFp4DstType(int64_t dst_type) { return dst_type == DST_TYPE_FP4_E2M1 || dst_type == DST_TYPE_FP4_E1M2; }
} // namespace

// Always uses V2 aclnn API. When x3 is not provided (nullopt), V2 internally
// behaves as V1 (x = x1 + x2). This eliminates the need for V1/V2 dispatch.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> add_rms_norm_dynamic_quant(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& gamma, const c10::optional<at::Tensor>& beta,
    const c10::optional<at::Tensor>& x3, double epsilon, int64_t scale_alg, c10::string_view round_mode,
    int64_t dst_type, bool output_rstd)
{
    TORCH_CHECK(x1.device().type() == at::kPrivateUse1, "x1 must be on NPU device, but got ", x1.device());
    TORCH_CHECK(x2.device().type() == at::kPrivateUse1, "x2 must be on NPU device, but got ", x2.device());
    TORCH_CHECK(gamma.device().type() == at::kPrivateUse1, "gamma must be on NPU device, but got ", gamma.device());

    TORCH_CHECK(x1.dim() >= 1 && x1.dim() <= 7, "x1 must be 1-7 dimensional, but got ", x1.dim());
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape, but got ", x1.sizes(), " vs ",
                x2.sizes());
    TORCH_CHECK(gamma.dim() == 1, "gamma must be 1-dimensional, but got ", gamma.dim());
    TORCH_CHECK(gamma.size(0) == x1.size(-1), "gamma size (", gamma.size(0), ") must match x1 last dimension (",
                x1.size(-1), ")");

    auto input_dtype = x1.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "x1 dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(x2.scalar_type() == input_dtype, "x2 dtype must match x1 dtype: ", input_dtype, " vs ",
                x2.scalar_type());

    if (beta.has_value() && beta->defined()) {
        TORCH_CHECK(beta->device().type() == at::kPrivateUse1, "beta must be on NPU device");
        TORCH_CHECK(beta->dim() == 1, "beta must be 1-dimensional, but got ", beta->dim());
        TORCH_CHECK(beta->size(0) == x1.size(-1), "beta size must match x1 last dimension");
    }
    if (x3.has_value() && x3->defined()) {
        TORCH_CHECK(x3->device().type() == at::kPrivateUse1, "x3 must be on NPU device");
        TORCH_CHECK(x3->sizes() == x1.sizes(), "x3 shape must match x1");
        TORCH_CHECK(x3->scalar_type() == input_dtype, "x3 dtype must match x1");
    }

    aclDataType y_acltype = GetAclDataTypeFromDstType(dst_type);
    at::ScalarType y_scalar = GetScalarTypeFromDstType(dst_type);
    if (IsFp4DstType(dst_type)) {
        TORCH_CHECK(x1.size(-1) % 2 == 0, "x1 last dim must be even for FP4 dst_type, but got ", x1.size(-1));
        TORCH_CHECK(scale_alg == 0, "scale_alg must be 0 (OCP) for FP4 dst_type, but got ", scale_alg);
    }
    char* round_mode_ptr = const_cast<char*>(round_mode.data());

    auto y_shape = x1.sizes().vec();
    if (IsFp4DstType(dst_type)) {
        y_shape.back() /= 2; // Two FP4 values packed per uint8
    }

    auto x_out_shape = x1.sizes().vec();

    // mxscale: rank = x1.rank + 1
    //   shape[-2] = ceil(ceil(x1.size(-1) / 32) / 2) = (num_blocks + 1) / 2
    //   shape[-1] = 2
    auto mxscale_shape = x1.sizes().vec();
    int64_t num_blocks = (x1.size(-1) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mxscale_shape.back() = (num_blocks + 1) / 2;
    mxscale_shape.emplace_back(MXSCALE_TAIL_DIM);

    // rstd: same rank as x1, norm dim set to 1
    auto rstd_shape = x1.sizes().vec();
    rstd_shape.back() = 1;

    // Output tensors use raw ND format (no ACL format).
    at::Tensor y = at::empty(y_shape, x1.options().dtype(y_scalar));
    at::Tensor x_out = at::empty(x_out_shape, x1.options());
    at::Tensor mxscale = at::empty(mxscale_shape, x1.options().dtype(at::ScalarType::Float8_e8m0fnu));
    at::Tensor rstd = at::empty(rstd_shape, x1.options().dtype(at::kFloat));

    TensorWrapper y_wrapper = {y, y_acltype};

    ACLNN_CMD(aclnnAddRmsNormDynamicMxQuantV2, x1, x2, gamma, beta, x3, epsilon, scale_alg, round_mode_ptr, dst_type,
              output_rstd, y_wrapper, x_out, mxscale, rstd);

    return std::make_tuple(std::move(y), std::move(x_out), std::move(mxscale), std::move(rstd));
}

} // namespace norm
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_rms_norm_dynamic_quant", &cann_ops_nn::norm::add_rms_norm_dynamic_quant,
          "AddRmsNormDynamicMxQuant fused operator on NPU (Add + RmsNorm + MX dynamic quantization)");
}
