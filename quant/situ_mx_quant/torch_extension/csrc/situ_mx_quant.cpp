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
// dst_type枚举值，与aclDataType保持一致：36=FLOAT8_E4M3FN，35=FLOAT8_E5M2
constexpr int64_t kDstTypeFloat8E4m3fn = 36;
constexpr int64_t kDstTypeFloat8E5m2 = 35;
// Situ激活gate/up拆分因子：x最后一维为2H，激活输出最后一维为H
constexpr int64_t kSplitFactor = 2;
// MX量化块大小：每32个元素共享一个scale
constexpr int64_t kMxBlockSize = 32;
// y_scale最后一维对齐数：每(align_num * block_size)=64个元素共享一组scale
constexpr int64_t kScaleAlignNum = 2;
// 量化轴固定为最后一维（aclnn接口当前仅支持-1）
constexpr int64_t kAxisLastDim = -1;

void CheckNpuTensor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(torch_npu::utils::is_npu(tensor), name, " must be on NPU device");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> situ_mx_quant(const at::Tensor& x, double beta, double linear_beta,
                                                 bool activate_left, int64_t dst_type, const std::string& round_mode)
{
    // 入参校验：设备/维度/数据类型/取值范围
    // 注意：空Tensor由aclnn侧正常处理（输出为空，见network测试M=0用例），非连续Tensor
    // aclnn亦支持，此处不做额外限制，与基线行为保持一致。
    CheckNpuTensor(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must be at least 1-dimensional, but got ", x.dim());
    const int64_t lastDim = x.size(x.dim() - 1);
    TORCH_CHECK(lastDim % kSplitFactor == 0, "x last dim must be even, but got ", lastDim);
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "x dtype must be float16 or bfloat16, but got ", x.scalar_type());
    TORCH_CHECK(beta > 0.0, "beta must be greater than 0, but got ", beta);
    TORCH_CHECK(dst_type == kDstTypeFloat8E4m3fn || dst_type == kDstTypeFloat8E5m2,
                "dst_type must be 36(E4M3FN) or 35(E5M2), but got ", dst_type);
    TORCH_CHECK(round_mode == "rint", "round_mode must be 'rint' for FP8 output, but got ", round_mode);

    // 输出y：shape与x一致，最后一维减半
    auto yShape = op_infer::array_to_small_vector(x.sizes());
    yShape[x.dim() - 1] = lastDim / kSplitFactor;

    // 输出y_scale：最后一维替换为ceil(H / 64)，再追加长度为kScaleAlignNum的维度
    constexpr int64_t scaleGroupSize = kScaleAlignNum * kMxBlockSize;
    auto yScaleShape = yShape;
    yScaleShape[x.dim() - 1] = (yShape[x.dim() - 1] + scaleGroupSize - 1) / scaleGroupSize;
    yScaleShape.push_back(kScaleAlignNum);

    // y的dtype由dst_type决定；y_scale为E8M0。两者均用at::empty按目标dtype直接申请：
    // NPU zero_算子不支持FP8/E8M0 dtype，禁止用at::zeros初始化FP8类输出；
    // ACLNN_CMD内部按scalar_type完成at::ScalarType->aclDataType的映射（E8M0->ACL_FLOAT8_E8M0）。
    const at::ScalarType yScalarType = (dst_type == kDstTypeFloat8E5m2) ? at::ScalarType::Float8_e5m2 :
                                                                          at::ScalarType::Float8_e4m3fn;
    const at::ScalarType yScaleScalarType = at::ScalarType::Float8_e8m0fnu;
    at::Tensor y = at::empty(yShape, x.options().dtype(yScalarType));
    at::Tensor yScale = at::empty(yScaleShape, x.options().dtype(yScaleScalarType));

    // 拉起aclnnSituMxQuant kernel，入参顺序与aclnn接口定义一致。
    // 注意：aclnn C接口的浮点标量原型为double（与本仓其他算子一致，勿按文档float截断，
    // 否则按ABI错位读参会导致beta/linear_beta变为非正规数，激活结果错误）。
    ACLNN_CMD(aclnnSituMxQuant, x, beta, linear_beta, activate_left, kAxisLastDim, dst_type, round_mode, y, yScale);

    return std::make_tuple(y, yScale);
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_mx_quant", &cann_ops_nn::quant::situ_mx_quant, "SituMxQuant on NPU");
}
