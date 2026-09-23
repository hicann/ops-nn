/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant_with_dual_axis.cpp
 * \brief Torch binding of the forward-only SwigluGroupQuantWithDualAxis operator.
 */

#include <vector>
#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn::activation {
namespace {
std::pair<c10::SymInt, c10::SymInt> CanonicalGeometry(const at::Tensor& x)
{
    // Output construction below indexes both axes before entering ACLNN.
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    return {x.sym_size(0), x.sym_size(1) / 2};
}

c10::SymInt CeilDiv(const c10::SymInt& value, int64_t factor)
{
    const c10::SymInt divisor(factor);
    const c10::SymInt remainder = value % divisor;
    return value / divisor + (remainder + c10::SymInt(factor - 1)) / divisor;
}

void CheckNpu(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined() && torch_npu::utils::is_npu(tensor), name, " must be a defined NPU tensor");
}

at::ScalarType GetOutputScalarType(aclDataType dtype)
{
    if (dtype == ACL_FLOAT8_E5M2) {
        return at::ScalarType::Float8_e5m2;
    }
    TORCH_CHECK(dtype == ACL_FLOAT8_E4M3FN, "dst_type must select FLOAT8_E5M2 or FLOAT8_E4M3FN");
    return at::ScalarType::Float8_e4m3fn;
}
} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant_with_dual_axis(
    const at::Tensor& x, const c10::optional<at::Tensor>& weight, const c10::optional<at::Tensor>& groupIndex,
    int64_t dstType, int64_t quantMode, double clampLimit, bool outputOrigin, double alpha, double bias)
{
    CheckNpu(x, "x");
    TORCH_CHECK(!x.requires_grad() && (!weight.has_value() || !weight->defined() || !weight->requires_grad()),
                "swiglu_group_quant_with_dual_axis is forward-only and does not support autograd");
    const aclDataType outputAclType = GetAclDataType(dstType);
    const auto outputScalarType = GetOutputScalarType(outputAclType);
    const auto [rows, cols] = CanonicalGeometry(x);
    if (weight.has_value() && weight->defined()) {
        CheckNpu(*weight, "weight");
    }

    c10::SymInt groups = 1;
    bool hasGroup = false;
    if (groupIndex.has_value() && groupIndex->defined()) {
        CheckNpu(*groupIndex, "group_index");
        groups = groupIndex->sym_numel();
        hasGroup = true;
    }

    auto yOptions = x.options().dtype(outputScalarType);
    auto scaleOptions = x.options().dtype(at::ScalarType::Float8_e8m0fnu);
    auto yShape = hasGroup ? c10::SymDimVector{rows, cols} : c10::SymDimVector(x.sym_sizes());
    yShape.back() = cols;
    auto scale1Shape = yShape;
    scale1Shape.back() = CeilDiv(cols, 64);
    scale1Shape.push_back(2);
    auto scale2Shape = yShape;
    // Avoid overflowing rows + 63 before the backend validates empty/invalid shapes.
    scale2Shape[0] = hasGroup ? rows / 64 + groups : CeilDiv(rows, 64);
    scale2Shape.push_back(2);
    at::Tensor y1 = at::empty_symint(yShape, yOptions);
    at::Tensor mxScale1 = at::empty_symint(scale1Shape, scaleOptions);
    at::Tensor y2 = at::empty_symint(yShape, yOptions);
    at::Tensor mxScale2 = at::empty_symint(scale2Shape, scaleOptions);
    const c10::SymDimVector emptyShape{0};
    at::Tensor yOrigin = outputOrigin ? at::empty_symint(yShape, x.options()) :
                                        at::empty_symint(emptyShape, x.options());

    TensorWrapper y1Wrapper{y1, outputAclType};
    TensorWrapper scale1Wrapper{mxScale1, ACL_FLOAT8_E8M0};
    TensorWrapper y2Wrapper{y2, outputAclType};
    TensorWrapper scale2Wrapper{mxScale2, ACL_FLOAT8_E8M0};
    TensorWrapper originWrapper{yOrigin, ConvertToAclDataType(yOrigin.scalar_type())};
    ACLNN_CMD(aclnnSwigluGroupQuantWithDualAxis, x, weight, groupIndex, outputAclType, quantMode, clampLimit,
              outputOrigin, alpha, bias, y1Wrapper, scale1Wrapper, y2Wrapper, scale2Wrapper, originWrapper);
    return {y1, mxScale1, y2, mxScale2, yOrigin};
}
} // namespace cann_ops_nn::activation

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_group_quant_with_dual_axis", &cann_ops_nn::activation::swiglu_group_quant_with_dual_axis,
          "SwigluGroupQuantWithDualAxis forward on NPU");
}
