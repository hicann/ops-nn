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

# PTA-facing dst_type follows the torch_npu int enum used by
# npu_dynamic_mx_quant_with_dual_axis:
#   23  = torch.float8_e5m2
#   24  = torch.float8_e4m3fn
#   296 = torch_npu.float4_e2m1fn_x2
#   297 = torch_npu.float4_e1m2fn_x2
# None is normalized to 24 (torch.float8_e4m3fn).
DEFAULT_DST_TYPE = 24
TORCH_DST_TYPE_FP8_E5M2 = 23
TORCH_DST_TYPE_FP8_E4M3FN = 24
TORCH_NPU_DST_TYPE_FP4_E2M1 = 296
TORCH_NPU_DST_TYPE_FP4_E1M2 = 297
FP8_ACL_DST_TYPES = (35, 36)
FP4_ACL_DST_TYPES = (40, 41)
SUPPORTED_ACL_DST_TYPES = FP8_ACL_DST_TYPES + FP4_ACL_DST_TYPES
_TORCH_DST_TYPE_TO_ACL = {
    TORCH_DST_TYPE_FP8_E5M2: 35,
    TORCH_DST_TYPE_FP8_E4M3FN: 36,
    TORCH_NPU_DST_TYPE_FP4_E2M1: 40,
    TORCH_NPU_DST_TYPE_FP4_E1M2: 41,
}


def _resolve_dst_type(dst_type):
    """Normalize the PTA int dst_type to aclnn dst_type 35/36/40/41."""
    if dst_type is None:
        dst_type = DEFAULT_DST_TYPE
    if dst_type not in _TORCH_DST_TYPE_TO_ACL:
        raise ValueError(
            "dst_type must be one of "
            "23 (torch.float8_e5m2), 24 (torch.float8_e4m3fn), "
            "296 (torch_npu.float4_e2m1fn_x2), 297 (torch_npu.float4_e1m2fn_x2), "
            f"or None, but got {dst_type}"
        )
    return _TORCH_DST_TYPE_TO_ACL[dst_type]


def _dst_output_dtype(acl_dst_type):
    if acl_dst_type == 35:
        return torch.float8_e5m2
    if acl_dst_type == 36:
        return torch.float8_e4m3fn
    # Torch exposes packed FP4 as uint8 (2 values/byte).
    return torch.uint8


class ClaGateQuantOpBuilder(OpBuilder):
    """ClaGateQuant 四输入 Torch Extension 构建器。"""

    def __init__(self):
        super().__init__("cla_gate_quant")

    def sources(self) -> list:
        return [self.resolve_source("cla_gate_quant.cpp")]

    def schema(self) -> str:
        return (
            "cla_gate_quant("
            "Tensor global_attn, "
            "Tensor local_attn, "
            "Tensor global_gate_logits, "
            "Tensor local_gate_logits, "
            "*, "
            "int? dst_type=None, "
            "str round_mode='rint', "
            "int scale_alg=1, "
            "str input_attn_layout='TND', "
            "bool dual_axis_flag=False"
            ") -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def cla_gate_quant_meta(
            global_attn: torch.Tensor,
            local_attn: torch.Tensor,
            global_gate_logits: torch.Tensor,
            local_gate_logits: torch.Tensor,
            *,
            dst_type=None,
            round_mode: str = "rint",
            scale_alg: int = 1,
            input_attn_layout: str = "TND",
            dual_axis_flag: bool = False,
        ):
            dst_type = _resolve_dst_type(dst_type)
            torch._check(
                global_attn.dim() == 3,
                lambda: f"global_attn must be 3D [T,N,D], but got {global_attn.dim()}D",
            )
            torch._check(
                local_attn.dim() == 3,
                lambda: f"local_attn must be 3D [T,N,D], but got {local_attn.dim()}D",
            )
            torch._check(
                local_attn.shape == global_attn.shape,
                lambda: f"local_attn shape {tuple(local_attn.shape)} must match global_attn {tuple(global_attn.shape)}",
            )
            t, n, d = global_attn.shape
            torch._check(1 <= n <= 128, lambda: f"N must be in [1,128], got {n}")
            torch._check(d in (128, 256), lambda: f"D must be 128 or 256, got {d}")

            gate_dim = global_gate_logits.dim()
            torch._check(
                gate_dim == 2,
                lambda: f"global_gate_logits must be 2D [T,N], but got {gate_dim}D",
            )
            torch._check(
                local_gate_logits.dim() == gate_dim,
                lambda: "local_gate_logits rank must match global_gate_logits",
            )
            torch._check(
                local_gate_logits.shape == global_gate_logits.shape,
                lambda: "local_gate_logits shape must match global_gate_logits",
            )
            torch._check(
                global_gate_logits.shape[0] == t and global_gate_logits.shape[1] == n,
                lambda: "gate logits shape must match [T,N]",
            )
            torch._check(
                round_mode in ("rint", "round", "floor"),
                lambda: f"round_mode must be rint/round/floor, got {round_mode}",
            )
            torch._check(
                scale_alg in (0, 1),
                lambda: f"scale_alg must be 0 or 1, but got {scale_alg}",
            )
            torch._check(
                input_attn_layout == "TND",
                lambda: f"input_attn_layout must be 'TND', but got {input_attn_layout}",
            )
            if dst_type in FP4_ACL_DST_TYPES:
                torch._check(
                    scale_alg == 0,
                    lambda: "FP4 only supports scale_alg=0",
                )
                k_check = n * d
                torch._check(
                    k_check % 4 == 0,
                    lambda: "FP4 requires K=N*D to be divisible by 4",
                )
            else:
                torch._check(
                    round_mode == "rint",
                    lambda: f"FP8 only supports round_mode=rint, got {round_mode}",
                )

            k = n * d
            if dst_type in FP4_ACL_DST_TYPES:
                # torch has no native FP4 dtype; expose packed uint8 (2 values/byte).
                data_dtype = torch.uint8
                data_cols = k // 2
            else:
                data_dtype = _dst_output_dtype(dst_type)
                data_cols = k

            row_data = torch.empty([t, data_cols], dtype=data_dtype, device="meta")
            row_scale = torch.empty(
                [t, (k + 63) // 64, 2], dtype=torch.uint8, device="meta"
            )
            if not dual_axis_flag:
                col_data = torch.empty([0], dtype=data_dtype, device="meta")
                col_scale = torch.empty([0], dtype=torch.uint8, device="meta")
            else:
                col_data = torch.empty([t, data_cols], dtype=data_dtype, device="meta")
                col_scale = torch.empty(
                    [(t + 63) // 64, k, 2], dtype=torch.uint8, device="meta"
                )

            return row_data, row_scale, col_data, col_scale


cla_gate_quant_builder = ClaGateQuantOpBuilder()
cla_gate_quant_builder._ensure_initialized()


@impl(
    get_as_library(),
    cla_gate_quant_builder.name,
    "PrivateUse1",
)
def cla_gate_quant(
    global_attn: torch.Tensor,
    local_attn: torch.Tensor,
    global_gate_logits: torch.Tensor,
    local_gate_logits: torch.Tensor,
    *,
    dst_type=None,
    round_mode: str = "rint",
    scale_alg: int = 1,
    input_attn_layout: str = "TND",
    dual_axis_flag: bool = False,
):
    """NPU 上的四输入 ClaGateQuant 融合算子。"""
    dst_type = _resolve_dst_type(dst_type)
    op_module = cla_gate_quant_builder.load()
    return op_module.cla_gate_quant(
        global_attn,
        local_attn,
        global_gate_logits,
        local_gate_logits,
        round_mode,
        scale_alg,
        dst_type,
        input_attn_layout,
        dual_axis_flag,
    )
