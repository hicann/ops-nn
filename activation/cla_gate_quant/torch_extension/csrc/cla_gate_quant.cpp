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
#include <string>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
// aclnn/GE-facing dst_type values used by ClaGateQuant op.
constexpr int64_t ACL_DST_TYPE_FP8_E5M2 = 35;
constexpr int64_t ACL_DST_TYPE_FP8_E4M3FN = 36;
constexpr int64_t ACL_DST_TYPE_FP4_E2M1 = 40;
constexpr int64_t ACL_DST_TYPE_FP4_E1M2 = 41;

static aclDataType GetAclDataTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case ACL_DST_TYPE_FP8_E5M2:
            return ACL_FLOAT8_E5M2;
        case ACL_DST_TYPE_FP8_E4M3FN:
            return ACL_FLOAT8_E4M3FN;
        case ACL_DST_TYPE_FP4_E2M1:
            return ACL_FLOAT4_E2M1;
        case ACL_DST_TYPE_FP4_E1M2:
            return ACL_FLOAT4_E1M2;
        default:
            TORCH_CHECK(false, "unsupported normalized acl dst_type: ", dst_type);
    }
}

static at::ScalarType GetScalarTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case ACL_DST_TYPE_FP8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case ACL_DST_TYPE_FP8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        case ACL_DST_TYPE_FP4_E2M1:
        case ACL_DST_TYPE_FP4_E1M2:
            // Torch exposes packed FP4 as uint8 (2 values/byte).
            return at::kByte;
        default:
            TORCH_CHECK(false, "unsupported normalized acl dst_type: ", dst_type);
    }
}

static bool IsFp4DstType(int64_t acl_dst_type)
{
    return acl_dst_type == ACL_DST_TYPE_FP4_E2M1 || acl_dst_type == ACL_DST_TYPE_FP4_E1M2;
}

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().type() == at::kPrivateUse1, name, " must be on NPU device");
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> cla_gate_quant(
    const at::Tensor& global_attn, const at::Tensor& local_attn, const at::Tensor& global_gate_logits,
    const at::Tensor& local_gate_logits, const std::string& round_mode, int64_t scale_alg, int64_t dst_type,
    const std::string& input_attn_layout, bool dual_axis_flag)
{
    CheckNpuTensor(global_attn, "global_attn");
    CheckNpuTensor(local_attn, "local_attn");
    CheckNpuTensor(global_gate_logits, "global_gate_logits");
    CheckNpuTensor(local_gate_logits, "local_gate_logits");

    TORCH_CHECK(global_attn.device() == local_attn.device(), "global_attn and local_attn must be on the same device");
    TORCH_CHECK(global_attn.device() == global_gate_logits.device(),
                "global_attn and global_gate_logits must be on the same device");
    TORCH_CHECK(global_attn.device() == local_gate_logits.device(),
                "global_attn and local_gate_logits must be on the same device");

    TORCH_CHECK(global_attn.dim() == 3, "global_attn must be 3D [T,N,D], but got ", global_attn.dim(), "D");
    TORCH_CHECK(local_attn.dim() == 3, "local_attn must be 3D [T,N,D], but got ", local_attn.dim(), "D");
    TORCH_CHECK(local_attn.sizes() == global_attn.sizes(), "local_attn shape must match global_attn shape");

    const int64_t t = global_attn.size(0);
    const int64_t n = global_attn.size(1);
    const int64_t d = global_attn.size(2);
    TORCH_CHECK(n >= 1 && n <= 128, "N must be in [1,128], got ", n);
    TORCH_CHECK(d == 128 || d == 256, "D must be 128 or 256, got ", d);

    const int64_t gate_dim = global_gate_logits.dim();
    TORCH_CHECK(gate_dim == 2, "global_gate_logits must be [T,N], but got ", gate_dim, "D");
    TORCH_CHECK(local_gate_logits.dim() == gate_dim, "local_gate_logits rank must match global_gate_logits rank");
    TORCH_CHECK(local_gate_logits.sizes() == global_gate_logits.sizes(),
                "local_gate_logits shape must match global_gate_logits shape");
    TORCH_CHECK(global_gate_logits.size(0) == t && global_gate_logits.size(1) == n, "gate logits must match [T,N]");

    const auto input_dtype = global_attn.scalar_type();
    TORCH_CHECK(input_dtype == at::kHalf || input_dtype == at::kBFloat16,
                "global_attn dtype must be float16 or bfloat16, but got ", input_dtype);
    TORCH_CHECK(local_attn.scalar_type() == input_dtype, "local_attn dtype must match global_attn dtype");
    TORCH_CHECK(global_gate_logits.scalar_type() == input_dtype,
                "global_gate_logits dtype must match global_attn dtype");
    TORCH_CHECK(local_gate_logits.scalar_type() == input_dtype, "local_gate_logits dtype must match global_attn dtype");

    TORCH_CHECK(round_mode == "rint" || round_mode == "round" || round_mode == "floor",
                "round_mode must be rint/round/floor, got ", round_mode);
    TORCH_CHECK(scale_alg == 0 || scale_alg == 1, "scale_alg must be 0 or 1, but got ", scale_alg);
    TORCH_CHECK(input_attn_layout == "TND", "input_attn_layout must be 'TND', but got ", input_attn_layout);
    const int64_t acl_dst_type = dst_type;
    if (IsFp4DstType(acl_dst_type)) {
        TORCH_CHECK(scale_alg == 0, "FP4 only supports scale_alg=0");
        TORCH_CHECK((n * d) % 4 == 0, "FP4 requires K=N*D to be divisible by 4");
    } else {
        TORCH_CHECK(acl_dst_type == ACL_DST_TYPE_FP8_E4M3FN || acl_dst_type == ACL_DST_TYPE_FP8_E5M2,
                    "dst_type must be 35/36(FP8) or 40/41(FP4), but got ", acl_dst_type);
        TORCH_CHECK(round_mode == "rint", "FP8 only supports round_mode=rint");
    }
    aclDataType data_acl_type = GetAclDataTypeFromDstType(acl_dst_type);
    at::ScalarType data_scalar_type = GetScalarTypeFromDstType(acl_dst_type);

    const int64_t k = n * d;
    const int64_t data_cols = IsFp4DstType(acl_dst_type) ? k / 2 : k;
    at::Tensor row_data = at::empty({t, data_cols}, global_attn.options().dtype(data_scalar_type));
    at::Tensor row_scale = at::empty({t, (k + SPLIT_BLOCK_SIZE - 1) / SPLIT_BLOCK_SIZE, 2},
                                     global_attn.options().dtype(at::kByte));
    at::Tensor col_data;
    at::Tensor col_scale;
    at::Tensor col_data_api;
    at::Tensor col_scale_api;
    if (!dual_axis_flag) {
        col_data = at::empty({0}, global_attn.options().dtype(data_scalar_type));
        col_scale = at::empty({0}, global_attn.options().dtype(at::kByte));
        // Single-axis mode: aclnn API requires col outputs to be nullptr.
        // Keep the returned tensors empty but do not pass them as aclTensor.
    } else {
        col_data = at::empty({t, data_cols}, global_attn.options().dtype(data_scalar_type));
        col_scale = at::empty({(t + SPLIT_BLOCK_SIZE - 1) / SPLIT_BLOCK_SIZE, k, 2},
                              global_attn.options().dtype(at::kByte));
        col_data_api = col_data;
        col_scale_api = col_scale;
    }

    TensorWrapper row_data_wrapper = {row_data, data_acl_type};
    TensorWrapper col_data_wrapper = {col_data_api, data_acl_type};
    TensorWrapper row_scale_wrapper = {row_scale, ACL_FLOAT8_E8M0};
    TensorWrapper col_scale_wrapper = {col_scale_api, ACL_FLOAT8_E8M0};

    // ACLNN_CMD binds its arguments to non-const lvalue references, so a temporary such as
    // std::string::c_str() cannot be passed. ConvertType() has an overload for const std::string&
    // that yields the const char* the aclnn API expects, so pass the strings as lvalues.
    ACLNN_CMD(aclnnClaGateQuant, global_attn, local_attn, global_gate_logits, local_gate_logits, round_mode, scale_alg,
              acl_dst_type, input_attn_layout, dual_axis_flag, row_data_wrapper, row_scale_wrapper, col_data_wrapper,
              col_scale_wrapper);

    return std::make_tuple(std::move(row_data), std::move(row_scale), std::move(col_data), std::move(col_scale));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("cla_gate_quant", &cla_gate_quant, "ClaGateQuant operator on NPU"); }
} // namespace op_api
