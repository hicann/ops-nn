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
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {

std::tuple<at::Tensor, at::Tensor> dequant_situ_quant(const at::Tensor& x, const c10::optional<at::Tensor>& weightScale,
                                                      const c10::optional<at::Tensor>& activationScale,
                                                      const c10::optional<at::Tensor>& bias,
                                                      const c10::optional<at::Tensor>& quantScale,
                                                      const c10::optional<at::Tensor>& quantOffset,
                                                      const c10::optional<at::Tensor>& groupIndex, double beta,
                                                      double linearBeta, bool activateLeft, std::string quantType)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on NPU device");

    const bool is_int8 = x.scalar_type() == at::kChar;
    const bool is_int32 = x.scalar_type() == at::kInt;
    const bool is_bf16 = x.scalar_type() == at::kBFloat16;
    const bool is_fp16 = x.scalar_type() == at::kHalf;
    TORCH_CHECK(is_int8 || is_int32 || is_bf16 || is_fp16,
                "x dtype must be int8, int32, bfloat16, or float16, but got ", x.scalar_type());

    TORCH_CHECK(x.size(-1) % 2 == 0, "x last dim must be even, but got ", x.size(-1));

    auto tensor_absent = [](const c10::optional<at::Tensor>& t) { return !t.has_value() || !t->defined(); };

    const at::Tensor weightScaleVal = weightScale.value_or(at::Tensor());
    const at::Tensor activationScaleVal = activationScale.value_or(at::Tensor());
    const at::Tensor biasVal = bias.value_or(at::Tensor());
    const at::Tensor quantScaleVal = quantScale.value_or(at::Tensor());
    const at::Tensor quantOffsetVal = quantOffset.value_or(at::Tensor());
    const at::Tensor groupIndexVal = groupIndex.value_or(at::Tensor());

    if (is_int8) {
        TORCH_CHECK(x.dim() >= 2, "x must be at least 2-dimensional for int8, but got ", x.dim());
        TORCH_CHECK(!tensor_absent(weightScale), "weight_scale is required for int8 x");
        TORCH_CHECK(tensor_absent(activationScale), "activation_scale must be absent for int8 x");
        TORCH_CHECK(tensor_absent(groupIndex), "group_index must be absent for int8 x");
        TORCH_CHECK(weightScaleVal.scalar_type() == at::kFloat, "weight_scale dtype must be float32, but got ",
                    weightScaleVal.scalar_type());
    } else {
        TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional for int32/bfloat16/float16, but got ", x.dim());
        if (is_int32) {
            TORCH_CHECK(!tensor_absent(weightScale), "weight_scale is required for int32 x");
            TORCH_CHECK(!tensor_absent(activationScale), "activation_scale is required for int32 x");
            TORCH_CHECK(tensor_absent(quantScale), "quant_scale is not supported for int32 x");
            TORCH_CHECK(tensor_absent(quantOffset), "quant_offset is not supported for int32 x");
            TORCH_CHECK(weightScaleVal.scalar_type() == at::kFloat, "weight_scale dtype must be float32, but got ",
                        weightScaleVal.scalar_type());
            TORCH_CHECK(activationScaleVal.scalar_type() == at::kFloat,
                        "activation_scale dtype must be float32, but got ", activationScaleVal.scalar_type());
            if (!tensor_absent(bias)) {
                TORCH_CHECK(biasVal.scalar_type() == at::kFloat, "bias dtype must be float32, but got ",
                            biasVal.scalar_type());
            }
        } else {
            // BF16 and FP16: pre-dequantized path, all optional inputs must be absent
            TORCH_CHECK(tensor_absent(weightScale), "weight_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(tensor_absent(activationScale), "activation_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(tensor_absent(bias), "bias must be absent for bfloat16/float16 x");
            TORCH_CHECK(tensor_absent(quantScale), "quant_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(tensor_absent(quantOffset), "quant_offset must be absent for bfloat16/float16 x");
            TORCH_CHECK(tensor_absent(groupIndex), "group_index must be absent for bfloat16/float16 x");
        }
    }

    std::vector<int64_t> yShape(x.sizes().begin(), x.sizes().end());
    yShape[x.dim() - 1] = x.size(x.dim() - 1) / 2;

    std::vector<int64_t> scaleShape;
    if (is_int8) {
        scaleShape.assign(x.sizes().begin(), x.sizes().end() - 1);
    } else {
        scaleShape.push_back(x.size(0));
    }

    at::Tensor y = at::empty(yShape, x.options().dtype(at::kChar));
    at::Tensor y_scale = at::empty(scaleShape, x.options().dtype(at::kFloat));

    if (x.size(0) == 0) {
        return std::make_tuple(y, y_scale);
    }

    ACLNN_CMD(aclnnDequantSituQuant, x, weightScaleVal, activationScaleVal, biasVal, quantScaleVal, quantOffsetVal,
              groupIndexVal, beta, linearBeta, activateLeft, quantType, y, y_scale);

    return std::make_tuple(y, y_scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dequant_situ_quant", &cann_ops_nn::quant::dequant_situ_quant, "DequantSituQuant on NPU");
}
