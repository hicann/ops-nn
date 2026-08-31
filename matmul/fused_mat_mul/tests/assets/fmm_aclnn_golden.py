#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""ACLNN-level golden for FusedMatMul.

C: (x1, x2, bias, x3, fusedOpType, cubeMathType, y)
bias/x3 may be None.  fusedOpType is a str.  y is the output tensor.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util


def hf32_truncate_torch(tensor):
    """Convert fp32 torch tensor to HF32 via numpy."""
    import torch

    arr = tensor.detach().cpu().numpy()
    result = _util.hf32_truncate_np(arr)
    return torch.from_numpy(result.copy()).to(tensor.dtype)


def _apply_cube_precision(tensors, cube_math_type):
    """Apply HF32 truncation or USE_FP16 downgrade to fp32 inputs."""
    import torch

    if cube_math_type in (1, 3) and tensors[0].dtype == torch.float32:
        return [hf32_truncate_torch(t) for t in tensors]
    if cube_math_type == 2 and tensors[0].dtype == torch.float32:
        return [t.to(torch.float16) for t in tensors]
    return list(tensors)


def torch_fused_matmul_core(
    x1,
    x2,
    bias=None,
    x3=None,
    *,
    fused_op_type="",
    out_dtype,
    cube_math_type=None,
    alpha=1.0,
    beta=1.0,
):
    """Core fused matmul for aclnn goldens.

    Precision flow:
      * cube_math_type in (1,3) + fp32 -> HF32 truncation (11-bit mantissa)
      * cube_math_type == 2 + fp32 -> forced downgrade to fp16
      * bf16/fp16 inputs -> upcast to fp32 (mirrors NPU cube fp32 accumulator)
      * matmul + bias (if present)
      * fused op applied: relu/add/mul/gelu_erf/gelu_tanh/16cast32
      * result cast to out_dtype
    """
    import torch

    x1, x2 = _apply_cube_precision([x1, x2], cube_math_type)

    if x1.dtype in (torch.float16, torch.bfloat16):
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

    mm_out = torch.matmul(x1, x2)

    if bias is not None:
        mm_out = mm_out + bias.to(torch.float32)

    if fused_op_type in ("add", "mul") and out_dtype is not None:
        if fused_op_type == "add" and (alpha != 1.0 or beta != 1.0):
            mm_out = alpha * mm_out + beta * x3.to(torch.float32)
        else:
            mm_out = mm_out.to(out_dtype)
            if fused_op_type == "add":
                mm_out = mm_out + x3.to(out_dtype)
            else:
                mm_out = mm_out * x3.to(out_dtype)
    elif fused_op_type == "relu":
        mm_out = torch.clamp(mm_out, min=0)
    elif fused_op_type == "gelu_erf":
        mm_out = 0.5 * mm_out * (1.0 + torch.erf(mm_out / np.sqrt(2.0)))
    elif fused_op_type == "gelu_tanh":
        mm_out = (
            0.5
            * mm_out
            * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (mm_out + 0.044715 * mm_out**3)))
        )

    if out_dtype is not None:
        mm_out = mm_out.to(out_dtype)
    return mm_out


class AclnnFusedMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1, x2, bias=None, x3=None, fusedOpType="", cubeMathType=0, y=None, **kwargs
    ):
        """aclnnFusedMatmul: y = FUSED_OP(x1 @ x2 + bias, x3)."""
        out_dtype = y.dtype if y is not None else x1.dtype
        return torch_fused_matmul_core(
            x1,
            x2,
            bias,
            x3,
            fused_op_type=fusedOpType,
            out_dtype=out_dtype,
            cube_math_type=cubeMathType,
        )


class AclnnFusedMatmulV2TestSpec:
    @staticmethod
    def golden(
        x1,
        x2,
        bias=None,
        x3=None,
        alphaOptional=1.0,
        betaOptional=1.0,
        fusedOpType="",
        cubeMathType=0,
        y=None,
        **kwargs,
    ):
        """aclnnFusedMatmulV2: y = alpha*(x1@x2) + beta*x3 (scale_add via add + scales)."""
        out_dtype = y.dtype if y is not None else x1.dtype
        return torch_fused_matmul_core(
            x1,
            x2,
            bias,
            x3,
            fused_op_type=fusedOpType,
            out_dtype=out_dtype,
            cube_math_type=cubeMathType,
            alpha=float(alphaOptional) if alphaOptional is not None else 1.0,
            beta=float(betaOptional) if betaOptional is not None else 1.0,
        )


__spec__ = {
    "aclnnFusedMatmul": "AclnnFusedMatmulTestSpec",
    "aclnnFusedMatmulV2": "AclnnFusedMatmulV2TestSpec",
}
