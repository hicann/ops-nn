# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# GemmSyrk PTA (PyTorch Adapter) Python 前端
# 管理 JIT 编译(csrc/gemm_syrk.cpp)并把算子注册到 PyTorch Dispatcher。
# 注册后可通过 torch.ops.cann_ops_nn.gemm_syrk(a, c, alpha=..., beta=..., transpose_x=..., fill_mode=...) 调用。
# 参考: torch_extension/README.md「新增算子」+ docs/torchapi_gemm_syrk.md。

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class GemmSyrkOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("gemm_syrk")

    def sources(self) -> list:
        return [self.resolve_source("gemm_syrk.cpp")]

    def schema(self) -> str:
        # C = alpha * (A @ A^T) + beta * C（transpose_x=True 时 a 为转置 (k, m) 存储，
        # 计算 C = alpha * (A^T @ A) + beta * C），原地写回 c 并返回 c（b! 为原地标记）。
        return (
            "gemm_syrk(Tensor a, Tensor(b!) c, *, Scalar? alpha=None, Scalar? beta=None, "
            'bool transpose_x=False, str fill_mode="full") -> Tensor(b!)'
        )

    def register_meta(self):
        @impl(get_as_library(), "gemm_syrk", "Meta")
        def gemm_syrk_meta(
            a: torch.Tensor,
            c: torch.Tensor,
            *,
            alpha=None,
            beta=None,
            transpose_x=False,
            fill_mode="full",
        ):
            return c


gemm_syrk_builder = GemmSyrkOpBuilder()
gemm_syrk_builder._ensure_initialized()


@impl(get_as_library(), "gemm_syrk", "PrivateUse1")
def gemm_syrk(
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    alpha=None,
    beta=None,
    transpose_x=False,
    fill_mode="full",
):
    """Dispatcher 的 NPU 实现。PrivateUse1 是 NPU 后端分发键。"""
    op_module = gemm_syrk_builder.load()
    return op_module.gemm_syrk(a, c, alpha, beta, transpose_x, fill_mode)
