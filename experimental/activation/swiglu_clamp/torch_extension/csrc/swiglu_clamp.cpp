/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See
 * LICENSE in the root of the software repository for the full text of the License.
 */

// SwigluClamp PTA (PyTorch Adapter) C++ 后端
// 把 PyTorch at::Tensor 桥接到 aclnnSwigluClamp(由算子本体 op_api/aclnn_swiglu_clamp.cpp 提供)。
// 参考: torch_extension/README.md「新增算子」+ PR !5910 (commit d8af5e890)。
//
// 与 README 模板的差异: 不 #include "aclnnop/aclnn_swiglu_clamp.h"。
//   ACLNN_CMD 宏(aclnn_common.h)用 #aclnn_api 字符串化 + dlsym 运行时解析,不依赖头里的函数声明;
//   且 swiglu_clamp 是 experimental 算子,头装在 custom_opp(<vendor>/op_api/include)而非 cann 标准 include,
//   硬 include 会令 JIT 找不到头。故省略,JIT 仅依赖框架 aclnn_common.h + cann 自带头。
//   运行期: 算子本体编译安装后,libcust_opapi.so 提供 aclnnSwigluClamp 符号(dlsym 解析)。

#include <torch/extension.h>
#include "aclnn_common.h" // ACLNN_CMD 宏(at::Tensor→aclTensor* + dlsym 调 aclnnSwigluClamp);靠 -I cann_ops_nn/common 解析,与 swiglu_group 等一致,不依赖 cpp 相对位置(适配 per-op 目录布局)

namespace cann_ops_nn {
namespace activation {

// SwigluClamp: out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
//   gate = x[..., :N], up = x[..., N:], 输出 shape [..., N]
at::Tensor swiglu_clamp(const at::Tensor& x, double limit)
{
    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "swiglu_clamp: x must be on NPU device");
    TORCH_CHECK(x.dim() >= 1, "swiglu_clamp: x must have at least 1 dim, got ", x.dim());
    TORCH_CHECK(x.size(-1) % 2 == 0, "swiglu_clamp: last dim must be even (2N), got ", x.size(-1));
    TORCH_CHECK(limit > 0, "swiglu_clamp: limit must be positive, got ", limit);

    // 输出 shape: 末维减半 [..., N]
    std::vector<int64_t> out_shape = x.sizes().vec();
    out_shape.back() /= 2;
    at::Tensor out = at::empty(out_shape, at::TensorOptions().dtype(x.dtype()).device(x.device()));

    // ACLNN_CMD 内部: 找 aclnnSwigluClamp{GetWorkspaceSize} 符号 → ConvertTypes 转参
    // → 调 GetWorkspaceSize(x→aclTensor*, limit→double, out→aclTensor*, ws, exec) → 调 aclnnSwigluClamp
    ACLNN_CMD(aclnnSwigluClamp, x, limit, out);
    return out;
}

} // namespace activation
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("swiglu_clamp", &cann_ops_nn::activation::swiglu_clamp,
          "SwigluClamp (silu-then-clamp fused activation) on NPU");
}
