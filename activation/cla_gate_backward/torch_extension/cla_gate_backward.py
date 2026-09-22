# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_nn.op_builder import OpBuilder, get_as_library


class ClaGateBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("cla_gate_backward")

    def sources(self):
        return [self.resolve_source("cla_gate_backward.cpp")]

    def schema(self):
        return (
            "cla_gate_backward("
            "Tensor grad_merged, "
            "Tensor global_attn, "
            "Tensor local_attn, "
            "Tensor global_gate_logits, "
            "Tensor local_gate_logits, "
            "*, "
            "str input_attn_layout='TND'"
            ") -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def cla_gate_backward_meta(
            grad_merged: torch.Tensor,
            global_attn: torch.Tensor,
            local_attn: torch.Tensor,
            global_gate_logits: torch.Tensor,
            local_gate_logits: torch.Tensor,
            *,
            input_attn_layout: str = "TND",
        ):
            # 三路 TND 输入必须为 3D [T, N, D] 且 shape 一致
            torch._check(
                grad_merged.dim() == 3,
                lambda: f"grad_merged must be 3D [T,N,D], but got {grad_merged.dim()}D",
            )
            torch._check(
                global_attn.dim() == 3,
                lambda: f"global_attn must be 3D [T,N,D], but got {global_attn.dim()}D",
            )
            torch._check(
                local_attn.dim() == 3,
                lambda: f"local_attn must be 3D [T,N,D], but got {local_attn.dim()}D",
            )
            torch._check(
                grad_merged.shape == global_attn.shape,
                lambda: f"grad_merged {tuple(grad_merged.shape)} must match global_attn {tuple(global_attn.shape)}",
            )
            torch._check(
                local_attn.shape == global_attn.shape,
                lambda: f"local_attn shape {tuple(local_attn.shape)} must match global_attn {tuple(global_attn.shape)}",
            )

            # gate logits：2D [T, N]
            torch._check(
                global_gate_logits.dim() == 2,
                lambda: f"global_gate_logits must be 2D [T,N], but got {global_gate_logits.dim()}D",
            )
            torch._check(
                local_gate_logits.dim() == 2,
                lambda: f"local_gate_logits must be 2D [T,N], but got {local_gate_logits.dim()}D",
            )
            torch._check(
                local_gate_logits.shape == global_gate_logits.shape,
                lambda: "local_gate_logits shape must match global_gate_logits",
            )
            t, n = global_attn.shape[0], global_attn.shape[1]
            torch._check(
                global_gate_logits.shape[0] == t and global_gate_logits.shape[1] == n,
                lambda: "gate logits shape must match [T,N]",
            )

            # dtype 校验：5 路同 dtype（BF16/FP16）
            input_dtype = global_attn.dtype
            torch._check(
                input_dtype in (torch.float16, torch.bfloat16),
                lambda: f"global_attn dtype must be float16 or bfloat16, but got {input_dtype}",
            )
            for name, tensor in [
                ("grad_merged", grad_merged),
                ("local_attn", local_attn),
                ("global_gate_logits", global_gate_logits),
                ("local_gate_logits", local_gate_logits),
            ]:
                torch._check(
                    tensor.dtype == input_dtype,
                    lambda: f"{name} dtype must match global_attn dtype ({input_dtype})",
                )

            # 不支持空 Tensor
            torch._check(
                grad_merged.numel() > 0
                and global_attn.numel() > 0
                and local_attn.numel() > 0
                and global_gate_logits.numel() > 0
                and local_gate_logits.numel() > 0,
                lambda: "ClaGateBackward does not support empty tensor",
            )

            torch._check(
                input_attn_layout == "TND",
                lambda: f"input_attn_layout only supports 'TND', but got {input_attn_layout}",
            )

            # 高进高出：输出与对应输入同 shape/dtype。
            grad_global_attn_out = torch.empty(
                global_attn.size(), dtype=global_attn.dtype, device=global_attn.device
            )
            grad_local_attn_out = torch.empty(
                local_attn.size(), dtype=local_attn.dtype, device=local_attn.device
            )
            grad_global_gate_logits_out = torch.empty(
                global_gate_logits.size(),
                dtype=global_gate_logits.dtype,
                device=global_gate_logits.device,
            )
            grad_local_gate_logits_out = torch.empty(
                local_gate_logits.size(),
                dtype=local_gate_logits.dtype,
                device=local_gate_logits.device,
            )
            return (
                grad_global_attn_out,
                grad_local_attn_out,
                grad_global_gate_logits_out,
                grad_local_gate_logits_out,
            )


cla_gate_backward_builder = ClaGateBackwardOpBuilder()
cla_gate_backward_builder._ensure_initialized()


@impl(
    get_as_library(),
    cla_gate_backward_builder.name,
    "PrivateUse1",
)
def cla_gate_backward(
    grad_merged: torch.Tensor,
    global_attn: torch.Tensor,
    local_attn: torch.Tensor,
    global_gate_logits: torch.Tensor,
    local_gate_logits: torch.Tensor,
    *,
    input_attn_layout: str = "TND",
):
    """NPU 上的 ClaGateBackward 反向融合算子（高进高出，不量化）。"""
    op_module = cla_gate_backward_builder.load()
    return op_module.cla_gate_backward(
        grad_merged,
        global_attn,
        local_attn,
        global_gate_logits,
        local_gate_logits,
        input_attn_layout,
    )
