# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# SwigluClamp PTA Python 前端
# 管理 JIT 编译(csrc/swiglu_clamp.cpp)并把算子注册到 PyTorch Dispatcher。
# 注册后可通过 torch.ops.cann_ops_nn.swiglu_clamp(x, limit) 调用。
# 参考: torch_extension/README.md「新增算子」+ PR !5910 (commit d8af5e890)。

import torch
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library


class SwigluClampOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("swiglu_clamp")

    def sources(self):
        """C++ 源码路径(相对 cann_ops_nn 包根)。"""
        return [self.resolve_source("swiglu_clamp.cpp")]

    def schema(self) -> str:
        """PyTorch 算子签名。limit 默认 7.0(Step-3.7 专家路)。"""
        return "swiglu_clamp(Tensor x, float limit=7.0) -> Tensor"

    def register_meta(self):
        """Meta 实现: 输出 shape [..., N](末维减半)。对 Autograd/FakeTensor/graph capture 必需。"""

        @impl(get_as_library(), "swiglu_clamp", "Meta")
        def swiglu_clamp_meta(x: torch.Tensor, limit: float = 7.0):
            out_shape = list(x.shape)
            out_shape[-1] = out_shape[-1] // 2
            return torch.empty(out_shape, dtype=x.dtype, device="meta")


# 模块级实例化 + 主动触发初始化: 注册 schema + meta,使 torch.ops.cann_ops_nn.swiglu_clamp 可见。
# 框架 OpBuilder._ensure_initialized 默认延迟到 load() 才 define schema,但那时 torch.ops.cann_ops_nn.swiglu_clamp
# attribute 根本不存在、无法被调用(load 的触发点恰是该 attribute)→ 死循环。故 import 时主动触发一次:
# define schema + 注册 Meta(.so 的 JIT 编译仍延迟到首次调用,不在此处编译)。
builder = SwigluClampOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), "swiglu_clamp", "PrivateUse1")
def swiglu_clamp(x: torch.Tensor, limit: float = 7.0):
    """Dispatcher 的 NPU 实现。PrivateUse1 是 NPU 后端分发键。"""
    op_module = builder.load()  # JIT 编译/加载 csrc/swiglu_clamp.cpp
    return op_module.swiglu_clamp(x, limit)
