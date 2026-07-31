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
#include "aclnnop/aclnn_grouped_dynamic_mx_quant_with_dual_axis.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
constexpr int64_t X_DIM_NUM = 2;
constexpr int64_t GROUP_INDEX_DIM_NUM = 1;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
constexpr int64_t DST_TYPE_FP8_E5M2 = 23;
constexpr int64_t DST_TYPE_FP8_E4M3FN = 24;

static aclDataType GetAclDataTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return ACL_FLOAT8_E5M2;
        case DST_TYPE_FP8_E4M3FN:
            return ACL_FLOAT8_E4M3FN;
        default:
            TORCH_CHECK(false, "dst_type should be 23(float8_e5m2) or 24(float8_e4m3fn), but got ", dst_type);
    }
}

static at::ScalarType GetScalarTypeFromDstType(int64_t dst_type)
{
    switch (dst_type) {
        case DST_TYPE_FP8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case DST_TYPE_FP8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        default:
            TORCH_CHECK(false, "dst_type should be 23(float8_e5m2) or 24(float8_e4m3fn), but got ", dst_type);
    }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> grouped_dynamic_mx_quant_with_dual_axis(
    const at::Tensor& x, const at::Tensor& group_index, c10::string_view round_mode, int64_t scale_alg,
    int64_t dst_type, double dst_type_max)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "The input x must be on an NPU device, but got ", x.device());
    TORCH_CHECK(group_index.device().type() == at::kPrivateUse1,
                "The input group_index must be on an NPU device, but got ", group_index.device());
    TORCH_CHECK(x.device() == group_index.device(), "The inputs x and group_index must be on the same device, but got ",
                x.device(), " and ", group_index.device());
    TORCH_CHECK(x.dim() == X_DIM_NUM, "The input x should be 2D, but got ", x.dim(), "D");
    TORCH_CHECK(x.size(-1) > 0 && x.size(-1) % SPLIT_BLOCK_SIZE == 0,
                "The last dim of input x must be > 0 and divisible by 64, but got ", x.size(-1));
    TORCH_CHECK(group_index.dim() == GROUP_INDEX_DIM_NUM && group_index.numel() > 0,
                "The input group_index should be a non-empty 1D tensor, but got ", group_index.dim(), "D with ",
                group_index.numel(), " elements");
    TORCH_CHECK(round_mode == "rint", "round_mode only supports \"rint\", but got \"", round_mode, "\"");
    aclDataType y_acltype = GetAclDataTypeFromDstType(dst_type);
    at::ScalarType scalar_dtype = GetScalarTypeFromDstType(dst_type);

    auto y_shape = x.sizes().vec();

    auto mxscale1_shape = x.sizes().vec();
    mxscale1_shape[mxscale1_shape.size() - 1] = x.size(-1) / SPLIT_BLOCK_SIZE;
    mxscale1_shape.emplace_back(NUM_TWO);

    auto mxscale2_shape = x.sizes().vec();
    mxscale2_shape[mxscale2_shape.size() - 2] = x.size(-2) / SPLIT_BLOCK_SIZE + group_index.size(0);
    mxscale2_shape.emplace_back(NUM_TWO);

    at::Tensor y1_out = at::empty(y_shape, x.options().dtype(scalar_dtype));
    at::Tensor y2_out = at::empty(y_shape, x.options().dtype(scalar_dtype));
    at::Tensor mxscale1 = at::empty(mxscale1_shape, x.options().dtype(at::kByte));
    at::Tensor mxscale2 = at::empty(mxscale2_shape, x.options().dtype(at::kByte));

    TensorWrapper y1_out_wrapper = {y1_out, y_acltype};
    TensorWrapper y2_out_wrapper = {y2_out, y_acltype};
    TensorWrapper mxscale1_wrapper = {mxscale1, ACL_FLOAT8_E8M0};
    TensorWrapper mxscale2_wrapper = {mxscale2, ACL_FLOAT8_E8M0};

    int64_t y_acltype_value = static_cast<int64_t>(y_acltype);
    char* round_mode_ptr = const_cast<char*>(round_mode.data());
    ACLNN_CMD(aclnnGroupedDynamicMxQuantWithDualAxis, x, group_index, round_mode_ptr, scale_alg, y_acltype_value,
              dst_type_max, y1_out_wrapper, mxscale1_wrapper, y2_out_wrapper, mxscale2_wrapper);
    return std::make_tuple(std::move(y1_out), std::move(mxscale1), std::move(y2_out), std::move(mxscale2));
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_dynamic_mx_quant_with_dual_axis", &cann_ops_nn::quant::grouped_dynamic_mx_quant_with_dual_axis,
          "GroupedDynamicMxQuantWithDualAxis operator on NPU");
}
