/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include "aclnnop/aclnn_quant_max_v2.h"
#include "aclnn_common.h"

namespace cann_ops_nn {

constexpr int64_t MAX_GROUP_NUM = 384;
constexpr double DST_TYPE_MAX_LIMIT = 32768.0;
namespace quant {

// torch 侧入口 quant_max 的 aclnn 实现。
// 语义：y = round(x * scale)（scale 为乘性比例 = 常规量化 scale 的倒数）；
//       scale 不传（nullopt）时走动态模式，先算各组 amax 再以 amax/dstTypeMax 量化，
//       并输出 y_scale 供反量化（y ≈ x * y_scale^-1 的倒数关系由框架侧使用）。
// 模式选择：scale 有无决定 静态/动态；groupList 有无决定 张量/分组（无 groupList 即 G=1）。
// 返回三元组 (y, y_scale, amax)：
//   y        形状同 x，dtype 由 dstType 决定；dstType=34(HIFLOAT8) 时为 uint8 位模式。
//   y_scale  动态模式为 [num_groups] float32；静态模式返回形状 [0] 的占位张量。
//   amax     恒为 [num_groups]，dtype 同 x。
std::tuple<at::Tensor, c10::optional<at::Tensor>, at::Tensor> QuantMax(const at::Tensor& x,
                                                                       const c10::optional<at::Tensor>& scale,
                                                                       const c10::optional<at::Tensor>& groupList,
                                                                       c10::string_view roundMode, int64_t dstType,
                                                                       double dstTypeMax)
{
    TORCH_CHECK(x.defined() && x.device().type() == at::kPrivateUse1, "x must be on NPU");
    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 8, "x rank must be 2..8");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "unsupported x dtype");
    const bool hasScale = scale.has_value() && scale->defined();
    const bool hasGroup = groupList.has_value() && groupList->defined();
    const int64_t groups = hasGroup ? groupList->numel() : 1;
    TORCH_CHECK(groups >= 1 && groups <= MAX_GROUP_NUM, "group count must be 1..384");
    if (hasScale) {
        TORCH_CHECK(scale->device() == x.device() && scale->scalar_type() == at::kFloat && scale->dim() == 1 &&
                        scale->numel() == groups,
                    "scale must be float32 [num_groups] on the same NPU; values multiply x");
    }
    if (hasGroup) {
        TORCH_CHECK(groupList->device() == x.device() && groupList->scalar_type() == at::kLong && groupList->dim() == 1,
                    "groupList must be int64 [num_groups] on the same NPU");
        // 边界单调、范围和末尾行号由调用方保证，与 V1 一致；此处不引入同步 D2H 检查。
    }
    TORCH_CHECK(dstType == 34 || dstType == 35 || dstType == 36, "unsupported dstType");
    TORCH_CHECK(
        (dstType == 34 && (roundMode == "round" || roundMode == "hybrid")) || (dstType != 34 && roundMode == "rint"),
        "roundMode does not match dstType");
    TORCH_CHECK(dstTypeMax >= 0.0 && dstTypeMax <= DST_TYPE_MAX_LIMIT, "invalid dstTypeMax");

    const at::ScalarType yDtype = dstType == 34 ? at::kByte :
                                  dstType == 35 ? at::ScalarType::Float8_e5m2 :
                                                  at::ScalarType::Float8_e4m3fn;
    const aclDataType aclYDtype = dstType == 34 ? ACL_HIFLOAT8 : dstType == 35 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
    at::Tensor y = at::empty(x.sizes(), x.options().dtype(yDtype));
    // amax 为必选输出，形状为 [num_groups]。
    at::Tensor amax = at::empty({groups}, x.options());
    // 动态输出 [num_groups] 的 y_scale；静态 ACLNN 传 nullptr，对外返回形状 [0] 的占位张量，kernel 不写 y_scale。
    at::Tensor yScaleOut = hasScale ? at::empty({0}, x.options().dtype(at::kFloat)) :
                                      at::empty({groups}, x.options().dtype(at::kFloat));
    c10::optional<at::Tensor> yScaleArg = hasScale ? c10::nullopt : c10::optional<at::Tensor>(yScaleOut);
    TensorWrapper yWrapper = {y, aclYDtype};
    char* roundModePtr = const_cast<char*>(roundMode.data());
    float dstTypeMaxF32 = static_cast<float>(dstTypeMax);
    ACLNN_CMD(aclnnQuantMaxV2, x, scale, groupList, roundModePtr, dstType, dstTypeMaxF32, yWrapper, amax, yScaleArg);
    return std::make_tuple(std::move(y), std::move(yScaleOut), std::move(amax));
}

} // namespace quant
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_max", &cann_ops_nn::quant::QuantMax,
          "QuantMaxV2 quantization operator on NPU (static/dynamic per-group quant to FP8)");
}
