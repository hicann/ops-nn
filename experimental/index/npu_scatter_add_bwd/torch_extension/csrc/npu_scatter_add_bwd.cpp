/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

// NpuScatterAddBwd PTA (PyTorch Adapter) C++ 后端
// 把 PyTorch at::Tensor 桥接到 aclnnNpuScatterAddBwd(由算子本体 op_api/aclnn_npu_scatter_add_bwd.cpp 提供)。
//
// 与 README 模板的差异: 不 #include "aclnnop/aclnn_npu_scatter_add_bwd.h"。
//   ACLNN_CMD 宏(aclnn_common.h)用 #aclnn_api 字符串化 + dlsym 运行时解析, 不依赖头里的函数声明;
//   且 npu_scatter_add_bwd 是 experimental 算子, 头装在 custom_opp(<vendor>/op_api/include)而非 cann 标准 include,
//   硬 include 会令 JIT 找不到头。故省略, JIT 仅依赖框架 aclnn_common.h + cann 自带头。
//   运行期: 算子本体编译安装后, libcust_opapi.so 提供 aclnnNpuScatterAddBwd 符号(dlsym 解析)。

#include <torch/extension.h>
#include "aclnn_common.h" // ACLNN_CMD 宏(at::Tensor→aclTensor* + dlsym 调 aclnnNpuScatterAddBwd); 靠 -I cann_ops_nn/common 解析

namespace cann_ops_nn {
namespace index {
namespace {
constexpr int64_t kYGradDimNum = 2;
constexpr int64_t kXDimNum = 2;
constexpr int64_t kIndexDimNum = 1;
} // namespace

// NpuScatterAddBwd: NpuScatterAdd(带缩放因子散射累加, MoE token 聚合)的反向算子
//   x_grad[i, :] = y_grad[indices[i], :] * s[i]
//   s_grad[i] = sum_j y_grad[indices[i], j] * x[i, j]
//   y_grad: (D, H), x: (N, H), s: (N,), indices: (N,); 输出 x_grad: (N, H), s_grad: (N,)
std::tuple<at::Tensor, at::Tensor> npu_scatter_add_bwd(const at::Tensor& y_grad, const at::Tensor& x,
                                                       const at::Tensor& s, const at::Tensor& indices)
{
    TORCH_CHECK(y_grad.device().type() == at::kPrivateUse1, "npu_scatter_add_bwd: y_grad must be on NPU device");
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "npu_scatter_add_bwd: x must be on NPU device");
    TORCH_CHECK(s.device().type() == at::kPrivateUse1, "npu_scatter_add_bwd: s must be on NPU device");
    TORCH_CHECK(indices.device().type() == at::kPrivateUse1, "npu_scatter_add_bwd: indices must be on NPU device");

    TORCH_CHECK(y_grad.scalar_type() == at::kBFloat16 || y_grad.scalar_type() == at::kHalf,
                "npu_scatter_add_bwd: y_grad dtype must be bfloat16 or float16, got ", y_grad.scalar_type());
    TORCH_CHECK(x.scalar_type() == y_grad.scalar_type(),
                "npu_scatter_add_bwd: x dtype must be the same as y_grad, got ", x.scalar_type());
    TORCH_CHECK(s.scalar_type() == y_grad.scalar_type(),
                "npu_scatter_add_bwd: s dtype must be the same as y_grad, got ", s.scalar_type());
    TORCH_CHECK(indices.scalar_type() == at::kInt, "npu_scatter_add_bwd: indices dtype must be int32, got ",
                indices.scalar_type());

    TORCH_CHECK(y_grad.dim() == kYGradDimNum, "npu_scatter_add_bwd: y_grad should be a 2d tensor, but got ",
                y_grad.dim(), " dims.");
    TORCH_CHECK(x.dim() == kXDimNum, "npu_scatter_add_bwd: x should be a 2d tensor, but got ", x.dim(), " dims.");
    TORCH_CHECK(s.dim() == kIndexDimNum, "npu_scatter_add_bwd: s should be a 1d tensor, but got ", s.dim(), " dims.");
    TORCH_CHECK(indices.dim() == kIndexDimNum, "npu_scatter_add_bwd: indices should be a 1d tensor, but got ",
                indices.dim(), " dims.");

    TORCH_CHECK(y_grad.size(1) == x.size(1), "npu_scatter_add_bwd: y_grad's dim[1](", y_grad.size(1),
                ") and x's dim[1](", x.size(1), ") should be the same.");
    TORCH_CHECK(x.size(0) == s.size(0), "npu_scatter_add_bwd: x's dim[0](", x.size(0), ") and s's dim[0](", s.size(0),
                ") should be the same.");
    TORCH_CHECK(x.size(0) == indices.size(0), "npu_scatter_add_bwd: x's dim[0](", x.size(0), ") and indices' dim[0](",
                indices.size(0), ") should be the same.");

    // 输出: x_grad 与 x 同 shape, s_grad 与 s 同 shape
    at::Tensor x_grad = at::empty(x.sizes(), at::TensorOptions().dtype(x.dtype()).device(x.device()));
    at::Tensor s_grad = at::empty(s.sizes(), at::TensorOptions().dtype(s.dtype()).device(s.device()));

    // ACLNN_CMD 内部: 找 aclnnNpuScatterAddBwd{GetWorkspaceSize} 符号 → ConvertTypes 转参
    // → 调 GetWorkspaceSize(y_grad/x/s/indices/x_grad/s_grad → aclTensor*, ws, exec) → 调 aclnnNpuScatterAddBwd
    ACLNN_CMD(aclnnNpuScatterAddBwd, y_grad, x, s, indices, x_grad, s_grad);
    return std::make_tuple(x_grad, s_grad);
}

} // namespace index
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_scatter_add_bwd", &cann_ops_nn::index::npu_scatter_add_bwd,
          "NpuScatterAddBwd (backward of scaled scatter-add for MoE) on NPU");
}
