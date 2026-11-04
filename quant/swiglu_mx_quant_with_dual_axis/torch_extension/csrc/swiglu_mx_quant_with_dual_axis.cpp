/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_mx_quant_with_dual_axis.cpp
 * \brief Torch binding of the forward-only SwigluMxQuantWithDualAxis operator.
 */

#include <torch/extension.h>
#include "aclnnop/aclnn_swiglu_mx_quant_with_dual_axis.h"
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t ROW_BLOCK_SIZE = 64;
constexpr int64_t MXSCALE_TAIL_DIM = 2;
constexpr int64_t FP4_PER_BYTE = 2;
constexpr int64_t DST_TYPE_FP8_E5M2 = 35;
constexpr int64_t DST_TYPE_FP8_E4M3FN = 36;
constexpr int64_t DST_TYPE_FP4_E2M1 = 40;
constexpr int64_t DST_TYPE_FP4_E1M2 = 41;

bool IsFp4DstType(int64_t dstType) { return dstType == DST_TYPE_FP4_E2M1 || dstType == DST_TYPE_FP4_E1M2; }

aclDataType GetAclDataTypeFromDstType(int64_t dstType)
{
    switch (dstType) {
        case DST_TYPE_FP8_E5M2:
            return ACL_FLOAT8_E5M2;
        case DST_TYPE_FP8_E4M3FN:
            return ACL_FLOAT8_E4M3FN;
        case DST_TYPE_FP4_E2M1:
            return ACL_FLOAT4_E2M1;
        case DST_TYPE_FP4_E1M2:
            return ACL_FLOAT4_E1M2;
        default:
            TORCH_CHECK(false, "dst_type must be 35(FP8_E5M2), 36(FP8_E4M3FN), 40(FP4_E2M1) or 41(FP4_E1M2), but got ",
                        dstType);
    }
}

at::ScalarType GetScalarTypeFromDstType(int64_t dstType)
{
    switch (dstType) {
        case DST_TYPE_FP8_E5M2:
            return at::ScalarType::Float8_e5m2;
        case DST_TYPE_FP8_E4M3FN:
            return at::ScalarType::Float8_e4m3fn;
        default:
            return at::kByte; // FP4 packed as uint8
    }
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> swiglu_mx_quant_with_dual_axis(
    const at::Tensor& x, const c10::optional<at::Tensor>& groupIndex, bool activateLeft, c10::string_view roundMode,
    int64_t scaleAlg, int64_t dstType, double maxDtypeValue)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device, but got ", x.device());
    TORCH_CHECK(!x.requires_grad(), "swiglu_mx_quant_with_dual_axis is forward-only and does not support autograd");
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional, but got ", x.dim());
    TORCH_CHECK(x.size(1) % FP4_PER_BYTE == 0, "x last dim must be even, but got ", x.size(1));
    const auto inputDtype = x.scalar_type();
    TORCH_CHECK(inputDtype == at::kHalf || inputDtype == at::kBFloat16, "x dtype must be float16 or bfloat16, but got ",
                inputDtype);

    // NOTE: the checks here mirror register_meta() in
    // swiglu_mx_quant_with_dual_axis.py and must stay in lockstep with it
    // (same order, same accept/reject semantics); they sit behind the NPU
    // device check, so only the Meta copy is reachable without hardware.
    const aclDataType yAclType = GetAclDataTypeFromDstType(dstType);
    const at::ScalarType yScalarType = GetScalarTypeFromDstType(dstType);
    TORCH_CHECK(scaleAlg == 0 || scaleAlg == 1, "scale_alg must be 0 (OCP) or 1 (cuBLAS, FP8 only), but got ",
                scaleAlg);
    if (IsFp4DstType(dstType)) {
        TORCH_CHECK(scaleAlg == 0, "scale_alg must be 0 (OCP) for FP4 dst_type, but got ", scaleAlg);
        TORCH_CHECK(roundMode == "rint" || roundMode == "floor" || roundMode == "round",
                    "round_mode must be rint, floor or round for FP4 dst_type, but got ", roundMode);
    } else {
        TORCH_CHECK(roundMode == "rint", "round_mode must be rint for FP8 dst_type, but got ", roundMode);
    }

    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1) / FP4_PER_BYTE;
    int64_t groups = 0;
    if (groupIndex.has_value() && groupIndex->defined()) {
        TORCH_CHECK(groupIndex->device().type() == at::kPrivateUse1, "group_index must be on NPU device");
        TORCH_CHECK(groupIndex->dim() == 1, "group_index must be 1-dimensional, but got ", groupIndex->dim());
        TORCH_CHECK(groupIndex->scalar_type() == at::kLong || groupIndex->scalar_type() == at::kInt,
                    "group_index dtype must be int32 or int64, but got ", groupIndex->scalar_type());
        groups = groupIndex->numel();
    }

    // y1/y2: [M, N]; FP4 packs two elements per byte along the last dim.
    std::vector<int64_t> yShape{rows, IsFp4DstType(dstType) ? cols / FP4_PER_BYTE : cols};
    // mxScale1 pairs the -1 axis blocks: [M, ceil(ceil(N/32)/2), 2]
    const int64_t colBlocks = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::vector<int64_t> scale1Shape{rows, (colBlocks + 1) / FP4_PER_BYTE, MXSCALE_TAIL_DIM};
    // mxScale2 covers the -2 axis: grouped layout keeps one pair gap per group
    const int64_t scale2Rows = groups > 0 ? rows / ROW_BLOCK_SIZE + groups :
                                            (rows + ROW_BLOCK_SIZE - 1) / ROW_BLOCK_SIZE;
    std::vector<int64_t> scale2Shape{scale2Rows, cols, MXSCALE_TAIL_DIM};

    auto yOptions = x.options().dtype(yScalarType);
    auto scaleOptions = x.options().dtype(at::ScalarType::Float8_e8m0fnu);
    at::Tensor y1 = at::empty(yShape, yOptions);
    at::Tensor mxScale1 = at::empty(scale1Shape, scaleOptions);
    at::Tensor y2 = at::empty(yShape, yOptions);
    at::Tensor mxScale2 = at::empty(scale2Shape, scaleOptions);

    // FP4 y is packed along the last dim; declare packedDim so the wrapper
    // restores the logical element count instead of guessing from strides.
    TensorWrapper y1Wrapper{y1, yAclType, y1.dim() - 1};
    TensorWrapper y2Wrapper{y2, yAclType, y2.dim() - 1};
    char* roundModePtr = const_cast<char*>(roundMode.data());
    ACLNN_CMD(aclnnSwigluMxQuantWithDualAxis, x, groupIndex, activateLeft, roundModePtr, scaleAlg, dstType,
              maxDtypeValue, y1Wrapper, mxScale1, y2Wrapper, mxScale2);

    return std::make_tuple(std::move(y1), std::move(mxScale1), std::move(y2), std::move(mxScale2));
}
} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_mx_quant_with_dual_axis", &cann_ops_nn::quant::swiglu_mx_quant_with_dual_axis,
          "SwigluMxQuantWithDualAxis forward on NPU (SwiGLU + dual-axis MX dynamic quantization)");
}
