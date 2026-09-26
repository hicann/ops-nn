# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# NpuScatterAddBwd PTA Python 前端
# 管理 JIT 编译(csrc/npu_scatter_add_bwd.cpp)并把算子注册到 PyTorch Dispatcher。
# 注册后可通过 torch.ops.cann_ops_nn.npu_scatter_add_bwd(y_grad, x, s, indices) 调用, 返回 (x_grad, s_grad)。

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class NpuScatterAddBwdOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_scatter_add_bwd")

    def sources(self):
        """C++ 源码路径(相对 cann_ops_nn 包根)。"""
        return [self.resolve_source("npu_scatter_add_bwd.cpp")]

    def schema(self) -> str:
        """PyTorch 算子签名。多输出: 返回 (x_grad, s_grad)。"""
        return (
            "npu_scatter_add_bwd(Tensor y_grad, Tensor x, Tensor s, Tensor indices) "
            "-> (Tensor x_grad, Tensor s_grad)"
        )

    def register_meta(self):
        """Meta 实现: x_grad 与 x 同 shape, s_grad 与 s 同 shape。对 Autograd/FakeTensor/graph capture 必需。"""

        @impl(get_as_library(), "npu_scatter_add_bwd", "Meta")
        def npu_scatter_add_bwd_meta(
            y_grad: torch.Tensor,
            x: torch.Tensor,
            s: torch.Tensor,
            indices: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            torch._check(
                y_grad.dim() == 2,
                lambda: f"y_grad must be a 2d tensor (D, H), but got {y_grad.dim()} dims.",
            )
            torch._check(
                x.dim() == 2,
                lambda: f"x must be a 2d tensor (N, H), but got {x.dim()} dims.",
            )
            torch._check(
                s.dim() == 1,
                lambda: f"s must be a 1d tensor (N,), but got {s.dim()} dims.",
            )
            torch._check(
                indices.dim() == 1,
                lambda: f"indices must be a 1d tensor (N,), but got {indices.dim()} dims.",
            )
            torch._check(
                y_grad.dtype in (torch.bfloat16, torch.float16),
                lambda: f"y_grad dtype must be bfloat16 or float16, but got {y_grad.dtype}.",
            )
            torch._check(
                x.dtype == y_grad.dtype,
                lambda: f"x dtype must be the same as y_grad, but got {x.dtype}.",
            )
            torch._check(
                s.dtype == y_grad.dtype,
                lambda: f"s dtype must be the same as y_grad, but got {s.dtype}.",
            )
            torch._check(
                indices.dtype == torch.int32,
                lambda: f"indices dtype must be int32, but got {indices.dtype}.",
            )
            torch._check(
                y_grad.size(1) == x.size(1),
                lambda: f"y_grad's dim[1]({y_grad.size(1)}) and x's dim[1]({x.size(1)}) should be the same.",
            )
            torch._check(
                x.size(0) == s.size(0),
                lambda: f"x's dim[0]({x.size(0)}) and s's dim[0]({s.size(0)}) should be the same.",
            )
            torch._check(
                x.size(0) == indices.size(0),
                lambda: f"x's dim[0]({x.size(0)}) and indices' dim[0]({indices.size(0)}) should be the same.",
            )
            x_grad = torch.empty(x.shape, dtype=x.dtype, device="meta")
            s_grad = torch.empty(s.shape, dtype=s.dtype, device="meta")
            return x_grad, s_grad


# 模块级实例化 + 主动触发初始化: 注册 schema + meta, 使 torch.ops.cann_ops_nn.npu_scatter_add_bwd 可见。
# .so 的 JIT 编译仍延迟到首次调用, 不在此处编译。
builder = NpuScatterAddBwdOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), "npu_scatter_add_bwd", "PrivateUse1")
def npu_scatter_add_bwd(
    y_grad: torch.Tensor, x: torch.Tensor, s: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NpuScatterAddBwd: NpuScatterAdd(带缩放因子散射累加, MoE token 聚合)的反向算子。

    计算公式(对每个 i = 0, 1, ..., N-1):
        x_grad[i, :] = y_grad[indices[i], :] * s[i]
        s_grad[i] = sum_j y_grad[indices[i], j] * x[i, j]

    :param y_grad: 上游梯度张量, shape (D, H), dtype 为 bfloat16/float16
    :param x: 前向中的源张量, shape (N, H), dtype 与 y_grad 一致
    :param s: 前向中的逐行缩放因子, shape (N,), dtype 与 y_grad 一致
    :param indices: 目标行索引, shape (N,), dtype 为 int32, 取值范围 [0, D)
    :return: 二元组 (x_grad, s_grad), 分别为 x 的梯度 shape (N, H) 与 s 的梯度 shape (N,)
    """
    op_module = builder.load()  # JIT 编译/加载 csrc/npu_scatter_add_bwd.cpp
    return op_module.npu_scatter_add_bwd(y_grad, x, s, indices)
