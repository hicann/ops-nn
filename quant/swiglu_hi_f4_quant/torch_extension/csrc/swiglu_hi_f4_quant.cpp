/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>

#include <torch/extension.h>
#include "aclnn_common.h"

namespace cann_ops_nn {
namespace quant {
namespace {
// SwiGLU激活gate/up拆分因子：activate_dim维为2H，激活输出该维为H
constexpr int64_t kSplitFactor = 2;
// HIFLOAT4为4-bit浮点，每字节存储2个元素，y(uint8)最后一维为元素数的一半
constexpr int64_t kHiF4PerByte = 2;
// scale量化块大小：每64个元素共享一个scale
constexpr int64_t kQuantBlockSize = 64;
// scale输出尾部对齐维长度
constexpr int64_t kScaleTailDim = 1;
// aclDataType枚举值（与目标CANN版本GE types.h保持一致）：42=ACL_HIFLOAT4，43=ACL_HIFLOAT4_SCALE。
// 当前环境acl头文件尚未定义该枚举，以数值常量表达以保证可编译性。
constexpr aclDataType kAclHiFloat4 = static_cast<aclDataType>(42);
constexpr aclDataType kAclHiFloat4Scale = static_cast<aclDataType>(43);

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}

void CheckOptionalNpuTensor(const c10::optional<at::Tensor>& tensor, const char* name)
{
    if (tensor.has_value() && tensor.value().defined()) {
        CheckNpuTensor(tensor.value(), name);
    }
}

int64_t CeilDiv(int64_t value, int64_t factor)
{
    TORCH_CHECK(factor > 0, "factor must be positive");
    return (value + factor - 1) / factor;
}
} // namespace

std::tuple<at::Tensor, at::Tensor> swiglu_hi_f4_quant(const at::Tensor& x, const c10::optional<at::Tensor>& group_index,
                                                      int64_t activate_dim, bool activate_left, int64_t swiglu_mode,
                                                      double clamp_limit, double glu_alpha, double glu_bias,
                                                      int64_t axis, int64_t scale_alg)
{
    // 入参校验：设备/维度/取值范围
    CheckNpuTensor(x, "x");
    CheckOptionalNpuTensor(group_index, "group_index");
    TORCH_CHECK(x.dim() > 1, "x dim should be greater than 1, but got ", x.dim());
    TORCH_CHECK(swiglu_mode == 0 || swiglu_mode == 1, "swiglu_mode only support 0 or 1, but got ", swiglu_mode);
    TORCH_CHECK(scale_alg == 0 || scale_alg == 1, "scale_alg only support 0 or 1, but got ", scale_alg);
    TORCH_CHECK(std::isfinite(clamp_limit) && clamp_limit > 0.0, "clamp_limit should be positive finite, but got ",
                clamp_limit);
    TORCH_CHECK(std::isfinite(glu_alpha), "glu_alpha should be finite, but got ", glu_alpha);
    TORCH_CHECK(std::isfinite(glu_bias), "glu_bias should be finite, but got ", glu_bias);

    // 归一化激活维与量化维（负值按倒数第N维解释），并校验范围
    int64_t activateDimValue = activate_dim;
    if (activateDimValue < 0) {
        activateDimValue = activateDimValue + x.dim();
    }
    TORCH_CHECK(activateDimValue >= 0 && activateDimValue <= (x.dim() - 1),
                "activate_dim should be in range [0, x.dim()-1], but got ", activate_dim);

    int64_t quantDimValue = axis;
    if (quantDimValue < 0) {
        quantDimValue = quantDimValue + x.dim();
    }
    TORCH_CHECK(quantDimValue >= 0 && quantDimValue <= (x.dim() - 1),
                "axis should be in range [0, x.dim()-1], but got ", axis);

    // 输出y：activate_dim维减半（SwiGLU门控拼接），最后一维再减半（HIFLOAT4每字节2个元素）
    // 输出scale：activate_dim维减半，量化维替换为scale组数，末尾追加长度1的对齐维
    c10::SmallVector<int64_t, op_infer::SIZE> ySize;
    c10::SmallVector<int64_t, op_infer::SIZE> scaleSize;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i == activateDimValue) {
            ySize.push_back(x.size(i) / kSplitFactor);
            scaleSize.push_back(x.size(i) / kSplitFactor);
        } else {
            ySize.push_back(x.size(i));
            scaleSize.push_back(x.size(i));
        }
    }
    ySize[x.dim() - 1] = ySize[x.dim() - 1] / kHiF4PerByte;

    // 量化维scale组数：
    // - 未传入group_index，或量化维为最后一维：ceil(quant_dim_size / 64)
    // - 传入group_index且量化维非最后一维：floor(quant_dim_size / 64) + group_index[0]（组边界补偿）
    int64_t quantSize = 0;
    const bool hasGroupIndex = group_index.has_value() && group_index.value().defined();
    if (!hasGroupIndex || quantDimValue == (x.dim() - 1)) {
        quantSize = CeilDiv(scaleSize[quantDimValue], kQuantBlockSize);
    } else {
        quantSize = scaleSize[quantDimValue] / kQuantBlockSize + group_index.value().sizes()[0];
    }
    scaleSize[quantDimValue] = quantSize;
    scaleSize.push_back(kScaleTailDim);

    // y以uint8存储（HIFLOAT4按fp4x2打包），scale为float32（HIFLOAT4_SCALE），
    // 通过TensorWrapper按aclnn侧声明的ACL_HIFLOAT4/ACL_HIFLOAT4_SCALE类型下发
    at::Tensor y = at::empty(ySize, x.options().dtype(at::kByte));
    at::Tensor scale = at::empty(scaleSize, x.options().dtype(at::kFloat));

    TensorWrapper yWrapper{y, kAclHiFloat4};
    TensorWrapper scaleWrapper{scale, kAclHiFloat4Scale};
    ACLNN_CMD(aclnnSwigluHiF4Quant, x, group_index, activateDimValue, activate_left, swiglu_mode, clamp_limit,
              glu_alpha, glu_bias, axis, scale_alg, yWrapper, scaleWrapper);
    return std::make_tuple(y, scale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_hi_f4_quant", &cann_ops_nn::quant::swiglu_hi_f4_quant, "SwigluHiF4Quant on NPU");
}
