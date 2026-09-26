# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# NpuScatterAdd PTA Python 前端
# 管理 JIT 编译(csrc/npu_scatter_add.cpp)并把算子注册到 PyTorch Dispatcher。
# 注册后可通过 torch.ops.cann_ops_nn.npu_scatter_add(x, y, indices, sort_idx) 调用, 结果 in-place 累加到 y 并返回 y。

from typing import Optional

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class NpuScatterAddOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_scatter_add")

    def sources(self):
        """C++ 源码路径(相对 cann_ops_nn 包根)。"""
        return [self.resolve_source("npu_scatter_add.cpp")]

    def schema(self) -> str:
        """PyTorch 算子签名。y 为 in-place 输入输出(a!); s/valid_token_num 可选, use_high_precision 为属性。"""
        return (
            "npu_scatter_add(Tensor x, Tensor(a!) y, Tensor indices, Tensor sort_idx, Tensor? s=None, "
            "Tensor? valid_token_num=None, *, bool use_high_precision=False) -> Tensor(a!)"
        )

    def register_meta(self):
        """Meta 实现: in-place 算子直接返回 y 自身。对 Autograd/FakeTensor/graph capture 必需。"""

        @impl(get_as_library(), "npu_scatter_add", "Meta")
        def npu_scatter_add_meta(
            x: torch.Tensor,
            y: torch.Tensor,
            indices: torch.Tensor,
            sort_idx: torch.Tensor,
            s: Optional[torch.Tensor] = None,
            valid_token_num: Optional[torch.Tensor] = None,
            *,
            use_high_precision: bool = False,
        ) -> torch.Tensor:
            torch._check(
                x.dim() == 2,
                lambda: f"x must be a 2d tensor (S, H), but got {x.dim()} dims.",
            )
            torch._check(
                y.dim() == 2,
                lambda: f"y must be a 2d tensor (D, H), but got {y.dim()} dims.",
            )
            torch._check(
                indices.dim() == 1,
                lambda: f"indices must be a 1d tensor (S,), but got {indices.dim()} dims.",
            )
            torch._check(
                sort_idx.dim() == 1,
                lambda: f"sort_idx must be a 1d tensor (S,), but got {sort_idx.dim()} dims.",
            )
            torch._check(
                x.dtype in (torch.bfloat16, torch.float16),
                lambda: f"x dtype must be bfloat16 or float16, but got {x.dtype}.",
            )
            torch._check(
                y.dtype == x.dtype,
                lambda: f"y dtype must be the same as x, but got {y.dtype}.",
            )
            torch._check(
                indices.dtype == torch.int32,
                lambda: f"indices dtype must be int32, but got {indices.dtype}.",
            )
            torch._check(
                sort_idx.dtype == torch.int32,
                lambda: f"sort_idx dtype must be int32, but got {sort_idx.dtype}.",
            )
            torch._check(
                x.size(1) == y.size(1),
                lambda: f"x's dim[1]({x.size(1)}) and y's dim[1]({y.size(1)}) should be the same.",
            )
            torch._check(
                x.size(0) == indices.size(0),
                lambda: f"x's dim[0]({x.size(0)}) and indices' dim[0]({indices.size(0)}) should be the same.",
            )
            torch._check(
                x.size(0) == sort_idx.size(0),
                lambda: f"x's dim[0]({x.size(0)}) and sort_idx's dim[0]({sort_idx.size(0)}) "
                "should be the same.",
            )
            if s is not None:
                torch._check(
                    s.dim() == 1,
                    lambda: f"s must be a 1d tensor (S,), but got {s.dim()} dims.",
                )
                torch._check(
                    s.dtype == x.dtype,
                    lambda: f"s dtype must be the same as x, but got {s.dtype}.",
                )
                torch._check(
                    s.size(0) == x.size(0),
                    lambda: f"s's dim[0]({s.size(0)}) and x's dim[0]({x.size(0)}) should be the same.",
                )
            if valid_token_num is not None:
                torch._check(
                    valid_token_num.dim() == 1 and valid_token_num.numel() == 1,
                    lambda: "valid_token_num should be a 1d tensor with 1 element.",
                )
                torch._check(
                    valid_token_num.dtype == torch.int32,
                    lambda: f"valid_token_num dtype must be int32, but got {valid_token_num.dtype}.",
                )
            return y


# 模块级实例化 + 主动触发初始化: 注册 schema + meta, 使 torch.ops.cann_ops_nn.npu_scatter_add 可见。
# .so 的 JIT 编译仍延迟到首次调用, 不在此处编译。
builder = NpuScatterAddOpBuilder()
builder._ensure_initialized()


@impl(get_as_library(), "npu_scatter_add", "PrivateUse1")
def npu_scatter_add(
    x: torch.Tensor,
    y: torch.Tensor,
    indices: torch.Tensor,
    sort_idx: torch.Tensor,
    s: Optional[torch.Tensor] = None,
    valid_token_num: Optional[torch.Tensor] = None,
    *,
    use_high_precision: bool = False,
) -> torch.Tensor:
    """
    NpuScatterAdd: 基于预排序索引的带缩放因子散射累加(MoE token 聚合)。

    计算公式(对每个 i = 0, 1, ..., S-1, in-place 累加到 y):
        y[indices[i], :] += x[i, :] * s[i] (提供 s 时)
        y[indices[i], :] += x[i, :] (不提供 s 时)

    :param x: 源张量, shape (S, H), dtype 为 bfloat16/float16
    :param y: 目标张量, shape (D, H), dtype 与 x 一致, 计算结果 in-place 累加写入
    :param indices: 目标行索引, shape (S,), dtype 为 int32, 取值范围 [0, D)
    :param sort_idx: argsort(indices) 的结果, shape (S,), dtype 为 int32
    :param s: 可选的逐行缩放因子, shape (S,), dtype 与 x 一致
    :param valid_token_num: 可选的有效 token 数, shape (1,), dtype 为 int32, 提供时仅处理前 valid_token_num 个行
    :param use_high_precision: 是否使用高精度模式(FP32 下完成缩放和累加后再转回原精度), 默认 False
    :return: 累加后的 y(与输入 y 为同一张量)
    """
    op_module = builder.load()  # JIT 编译/加载 csrc/npu_scatter_add.cpp
    return op_module.npu_scatter_add(
        x, y, indices, sort_idx, s, valid_token_num, use_high_precision
    )
