/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

// NpuScatterAdd PTA (PyTorch Adapter) C++ 后端
// 把 PyTorch at::Tensor 桥接到 aclnnNpuScatterAdd(由算子本体 op_api/aclnn_npu_scatter_add.cpp 提供)。
//
// 与 README 模板的差异: 不 #include "aclnnop/aclnn_npu_scatter_add.h"。
//   ACLNN_CMD 宏(aclnn_common.h)用 #aclnn_api 字符串化 + dlsym 运行时解析, 不依赖头里的函数声明;
//   且 npu_scatter_add 是 experimental 算子, 头装在 custom_opp(<vendor>/op_api/include)而非 cann 标准 include,
//   硬 include 会令 JIT 找不到头。故省略, JIT 仅依赖框架 aclnn_common.h + cann 自带头。
//   运行期: 算子本体编译安装后, libcust_opapi.so 提供 aclnnNpuScatterAdd 符号(dlsym 解析)。

#include <torch/extension.h>
#include "aclnn_common.h" // ACLNN_CMD 宏(at::Tensor→aclTensor* + dlsym 调 aclnnNpuScatterAdd); 靠 -I cann_ops_nn/common 解析

namespace cann_ops_nn {
namespace index {
namespace {
constexpr int64_t kXDimNum = 2;
constexpr int64_t kYDimNum = 2;
constexpr int64_t kIndexDimNum = 1;
constexpr int64_t kValidTokenNumElements = 1;

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
} // namespace

// NpuScatterAdd: 基于预排序索引的带缩放因子散射累加(MoE token 聚合), 结果 in-place 累加写入 y 并返回 y
//   y[indices[i], :] += x[i, :] * s[i] (提供 s 时); 不提供 s 时不加权
//   x: (S, H), y: (D, H), s: (S,), indices: (S,), sort_idx: argsort(indices)
at::Tensor npu_scatter_add(const at::Tensor& x, const at::Tensor& y, const at::Tensor& indices,
                           const at::Tensor& sort_idx, const c10::optional<at::Tensor>& s,
                           const c10::optional<at::Tensor>& valid_token_num, bool use_high_precision)
{
    CheckNpuTensor(x, "x");
    CheckNpuTensor(y, "y");
    CheckNpuTensor(indices, "indices");
    CheckNpuTensor(sort_idx, "sort_idx");
    CheckOptionalNpuTensor(s, "s");
    CheckOptionalNpuTensor(valid_token_num, "valid_token_num");

    const bool hasS = s.has_value() && s.value().defined();
    const bool hasValidTokenNum = valid_token_num.has_value() && valid_token_num.value().defined();

    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "npu_scatter_add: x dtype must be bfloat16 or float16, got ", x.scalar_type());
    TORCH_CHECK(y.scalar_type() == x.scalar_type(), "npu_scatter_add: y dtype must be the same as x, got ",
                y.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::kInt, "npu_scatter_add: indices dtype must be int32, got ",
                indices.scalar_type());
    TORCH_CHECK(sort_idx.scalar_type() == at::kInt, "npu_scatter_add: sort_idx dtype must be int32, got ",
                sort_idx.scalar_type());
    if (hasS) {
        TORCH_CHECK(s.value().scalar_type() == x.scalar_type(), "npu_scatter_add: s dtype must be the same as x, got ",
                    s.value().scalar_type());
    }
    if (hasValidTokenNum) {
        TORCH_CHECK(valid_token_num.value().scalar_type() == at::kInt,
                    "npu_scatter_add: valid_token_num dtype must be int32, got ",
                    valid_token_num.value().scalar_type());
    }

    TORCH_CHECK(x.dim() == kXDimNum, "npu_scatter_add: x should be a 2d tensor, but got ", x.dim(), " dims.");
    TORCH_CHECK(y.dim() == kYDimNum, "npu_scatter_add: y should be a 2d tensor, but got ", y.dim(), " dims.");
    TORCH_CHECK(indices.dim() == kIndexDimNum, "npu_scatter_add: indices should be a 1d tensor, but got ",
                indices.dim(), " dims.");
    TORCH_CHECK(sort_idx.dim() == kIndexDimNum, "npu_scatter_add: sort_idx should be a 1d tensor, but got ",
                sort_idx.dim(), " dims.");
    if (hasS) {
        TORCH_CHECK(s.value().dim() == kIndexDimNum, "npu_scatter_add: s should be a 1d tensor, but got ",
                    s.value().dim(), " dims.");
        TORCH_CHECK(s.value().size(0) == x.size(0), "npu_scatter_add: s's dim[0](", s.value().size(0),
                    ") and x's dim[0](", x.size(0), ") should be the same.");
    }
    if (hasValidTokenNum) {
        TORCH_CHECK(
            valid_token_num.value().dim() == kIndexDimNum && valid_token_num.value().numel() == kValidTokenNumElements,
            "npu_scatter_add: valid_token_num should be a 1d tensor with 1 element.");
    }

    TORCH_CHECK(x.size(1) == y.size(1), "npu_scatter_add: x's dim[1](", x.size(1), ") and y's dim[1](", y.size(1),
                ") should be the same.");
    TORCH_CHECK(x.size(0) == indices.size(0), "npu_scatter_add: x's dim[0](", x.size(0), ") and indices' dim[0](",
                indices.size(0), ") should be the same.");
    TORCH_CHECK(x.size(0) == sort_idx.size(0), "npu_scatter_add: x's dim[0](", x.size(0), ") and sort_idx's dim[0](",
                sort_idx.size(0), ") should be the same.");

    // ACLNN_CMD 内部: 找 aclnnNpuScatterAdd{GetWorkspaceSize} 符号 → ConvertTypes 转参(s/valid_token_num 为空时转
    // nullptr) → 调 GetWorkspaceSize(x/y/s/indices/sort_idx/valid_token_num/use_high_precision, ws, exec) → 调
    // aclnnNpuScatterAdd 注意: ACLNN_CMD 参数顺序须与 aclnnNpuScatterAddGetWorkspaceSize 一致(s 位于 indices 之前)
    ACLNN_CMD(aclnnNpuScatterAdd, x, y, s, indices, sort_idx, valid_token_num, use_high_precision);
    return y;
}

} // namespace index
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_scatter_add", &cann_ops_nn::index::npu_scatter_add,
          "NpuScatterAdd (scaled scatter-add for MoE, in-place on y) on NPU");
}
