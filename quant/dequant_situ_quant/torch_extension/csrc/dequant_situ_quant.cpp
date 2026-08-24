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
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
// gate与up在最后一维拼接，输出最后一维 = x最后一维 / kSplitFactor
constexpr int64_t kSplitFactor = 2;
// int8路径的最低维度；int32/bfloat16/float16路径的固定维度
constexpr int64_t kMinDimNum = 2;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
}

bool TensorAbsent(const c10::optional<at::Tensor>& tensor) { return !tensor.has_value() || !tensor->defined(); }

c10::SmallVector<int64_t, op_infer::SIZE> GetOutputShape(const at::Tensor& x)
{
    auto y_shape = op_infer::array_to_small_vector(x.sizes());
    y_shape[x.dim() - 1] = x.size(x.dim() - 1) / kSplitFactor;
    return y_shape;
}

c10::SmallVector<int64_t, op_infer::SIZE> GetScaleShape(const at::Tensor& x, bool is_int8)
{
    c10::SmallVector<int64_t, op_infer::SIZE> scale_shape;
    if (is_int8) {
        // int8路径：逐token动态量化，scale shape为x去掉最后一维
        for (int64_t i = 0; i < x.dim() - 1; ++i) {
            scale_shape.emplace_back(x.size(i));
        }
    } else {
        // 其余路径：逐行动态量化，scale shape为[M]
        scale_shape.emplace_back(x.size(0));
    }
    return scale_shape;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> dequant_situ_quant(const at::Tensor& x,
                                                      const c10::optional<at::Tensor>& weight_scale,
                                                      const c10::optional<at::Tensor>& activation_scale,
                                                      const c10::optional<at::Tensor>& bias,
                                                      const c10::optional<at::Tensor>& quant_scale,
                                                      const c10::optional<at::Tensor>& quant_offset,
                                                      const c10::optional<at::Tensor>& group_index, double beta,
                                                      double linear_beta, bool activate_left, std::string quant_type)
{
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(weight_scale, "weight_scale");
    CheckOptionalNpuTensor(activation_scale, "activation_scale");
    CheckOptionalNpuTensor(bias, "bias");
    CheckOptionalNpuTensor(quant_scale, "quant_scale");
    CheckOptionalNpuTensor(quant_offset, "quant_offset");
    CheckOptionalNpuTensor(group_index, "group_index");

    const bool is_int8 = x.scalar_type() == at::kChar;
    const bool is_int32 = x.scalar_type() == at::kInt;
    const bool is_bf16 = x.scalar_type() == at::kBFloat16;
    const bool is_fp16 = x.scalar_type() == at::kHalf;
    TORCH_CHECK(is_int8 || is_int32 || is_bf16 || is_fp16,
                "x dtype must be int8, int32, bfloat16, or float16, but got ", x.scalar_type());

    TORCH_CHECK(x.size(-1) % kSplitFactor == 0, "x last dim must be even, but got ", x.size(-1));

    if (is_int8) {
        // INT8路径：per-channel反量化 + static/dynamic量化
        TORCH_CHECK(x.dim() >= kMinDimNum, "x must be at least 2-dimensional for int8, but got ", x.dim());
        TORCH_CHECK(!TensorAbsent(weight_scale), "weight_scale is required for int8 x");
        TORCH_CHECK(TensorAbsent(activation_scale), "activation_scale must be absent for int8 x");
        TORCH_CHECK(TensorAbsent(group_index), "group_index must be absent for int8 x");
        TORCH_CHECK(weight_scale->scalar_type() == at::kFloat, "weight_scale dtype must be float32, but got ",
                    weight_scale->scalar_type());
    } else {
        TORCH_CHECK(x.dim() == kMinDimNum, "x must be 2-dimensional for int32/bfloat16/float16, but got ", x.dim());
        if (is_int32) {
            // INT32路径：MoE分组matmul结果反量化 + 动态量化
            TORCH_CHECK(!TensorAbsent(weight_scale), "weight_scale is required for int32 x");
            TORCH_CHECK(!TensorAbsent(activation_scale), "activation_scale is required for int32 x");
            TORCH_CHECK(TensorAbsent(quant_scale), "quant_scale is not supported for int32 x");
            TORCH_CHECK(TensorAbsent(quant_offset), "quant_offset is not supported for int32 x");
            TORCH_CHECK(weight_scale->scalar_type() == at::kFloat, "weight_scale dtype must be float32, but got ",
                        weight_scale->scalar_type());
            TORCH_CHECK(activation_scale->scalar_type() == at::kFloat,
                        "activation_scale dtype must be float32, but got ", activation_scale->scalar_type());
            if (!TensorAbsent(bias)) {
                TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias dtype must be float32, but got ",
                            bias->scalar_type());
            }
        } else {
            // BF16/FP16路径：预反量化输入，所有可选输入必须为空
            TORCH_CHECK(TensorAbsent(weight_scale), "weight_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(TensorAbsent(activation_scale), "activation_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(TensorAbsent(bias), "bias must be absent for bfloat16/float16 x");
            TORCH_CHECK(TensorAbsent(quant_scale), "quant_scale must be absent for bfloat16/float16 x");
            TORCH_CHECK(TensorAbsent(quant_offset), "quant_offset must be absent for bfloat16/float16 x");
            TORCH_CHECK(TensorAbsent(group_index), "group_index must be absent for bfloat16/float16 x");
        }
    }

    // y and y_scale must be zero-initialized. The kernel does NOT write output
    // elements that fall outside a processed group: when group_index contains
    // values <= 0 (treated as 0 rows) or when the sum of group_index is less
    // than the total row count, those rows' y and y_scale GM slots are left
    // untouched. With at::empty, the unwritten buffer contains leftover NPU
    // memory (garbage), which mismatches the CPU golden's torch.zeros for the
    // same skipped rows. at::zeros guarantees 0 for all unwritten elements.
    at::Tensor y = at::zeros(GetOutputShape(x), x.options().dtype(at::kChar));
    at::Tensor y_scale = at::zeros(GetScaleShape(x, is_int8), x.options().dtype(at::kFloat));

    if (x.size(0) == 0) {
        return std::make_tuple(y, y_scale);
    }

    // 拉起aclnnDequantSituQuant，入参顺序与aclnn接口定义一致
    ACLNN_CMD(aclnnDequantSituQuant, x, weight_scale, activation_scale, bias, quant_scale, quant_offset, group_index,
              beta, linear_beta, activate_left, quant_type, y, y_scale);

    return std::make_tuple(y, y_scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("dequant_situ_quant", &cann_ops_nn::quant::dequant_situ_quant, "DequantSituQuant on NPU");
}
