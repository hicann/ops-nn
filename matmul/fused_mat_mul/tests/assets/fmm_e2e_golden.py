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
"""E2E-level golden for torch_npu.npu_fused_matmul."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
import matmul_golden_util as _util
from fmm_aclnn_golden import torch_fused_matmul_core

# HF32（hfloat32）用例：fp32 数据 + hf32 计算模式。torch 通路无 per-call
# cubeMathType 参数，需进程级开关 ALLOW_MATMUL_HF32（torch.npu.matmul.allow_hf32）
# 驱动 op-plugin 传 USE_HF32（否则 aclnnFusedMatmul 拒绝 fp32）；
# golden 侧同步按 cmt=3 做 hf32 截断建模。开关只影响 fp32，套件内其余
# fp16/bf16 用例为 no-op（CheckCubeMathTypeForMm 对 BF16+HF32 仅告警）。
_USE_HF32_FOR_FP32 = True


def _enable_hf32_for_fp32(*tensors):
    """fp32 输入时开启进程级 hf32 开关，返回 golden 应使用的 cube_math_type。"""
    if not _USE_HF32_FOR_FP32:
        return None
    if not any(t is not None and t.dtype == torch.float32 for t in tensors):
        return None
    try:
        import torch_npu  # noqa: F401

        torch.npu.matmul.allow_hf32 = True
    except Exception:  # noqa: BLE001  (CPU/离线环境无 torch_npu)
        pass
    return 3


def _compat_impl_cmt(x1, x2, x3, fused_op_type):
    """兼容模式高精度判定（TORCH_NPU_USE_COMPATIBLE_IMPL=1）：fusedOpType 为
    add/mul、x1 为 fp16/bf16、x1/x2/x3 同 dtype 时传 USE_FP32_ADD(4)，对齐
    op-plugin（npu_fused_matmul）与 cann_ops wrapper（NeedFusedAddMulHighPrecision）
    的兼容语义；aclnn 侧 GetInnerPrecise 命中后 NPU 在 fp32 域融合（中间不
    round）。dim 的 2D/3D 对齐由 torch_fused_matmul_core 内的
    _use_fp32_domain_fusion 完成。"""
    if os.environ.get("TORCH_NPU_USE_COMPATIBLE_IMPL") != "1":
        return None
    if fused_op_type not in ("add", "mul") or x3 is None:
        return None
    if x1.dtype not in (torch.float16, torch.bfloat16):
        return None
    if x2.dtype != x1.dtype or x3.dtype != x1.dtype:
        return None
    return 4


def _cann_ops_cmt(x1, x2, x3, fused_op_type):
    """镜像 cann_ops wrapper 的 cmt 链路（fused_mat_mul.py 的 _get_cube_math_type
    + csrc/fused_matmul.cpp 的 NeedFusedAddMulHighPrecision 无条件覆盖）：
    1. 兼容模式命中 -> 4（最高优先级，覆盖显式配置）
    2. 显式 torch_npu.npu.matmul.cube_math_type -> int(cmt)
    3. torch.npu.matmul.allow_hf32 -> 3（USE_HF32）
    4. 默认 -> 0（KEEP_DTYPE）"""
    cmt = _compat_impl_cmt(x1, x2, x3, fused_op_type)
    if cmt is not None:
        return cmt
    try:
        import torch_npu

        explicit = torch_npu.npu.matmul.cube_math_type
        if explicit is not None:
            return int(explicit)
        if torch.npu.matmul.allow_hf32:
            return 3
    except Exception:  # noqa: BLE001  (CPU/离线环境无 torch_npu)
        pass
    return 0


class TorchNpuFusedMatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def customize_inputs(x1, x2, *, bias=None, x3=None, fused_op_type="", **kwargs):
        """执行前设置 hf32 开关：torch 通路无 per-call cubeMathType，只能靠进程级
        ALLOW_MATMUL_HF32 让 op-plugin 传 USE_HF32；该钩子在设备执行前被调用。"""
        _enable_hf32_for_fp32(x1, x2, bias, x3)

    @staticmethod
    def golden(x1, x2, *, bias=None, x3=None, fused_op_type="", **kwargs):
        """torch_npu.npu_fused_matmul: y = FUSED_OP(x1 @ x2 + bias, x3)."""
        out_dtype = torch.float32 if fused_op_type == "16cast32" else x1.dtype
        cube_math_type = _enable_hf32_for_fp32(x1, x2, bias, x3)
        if cube_math_type is None:
            # fp32 已按 hf32(3) 建模；低精度 add/mul 在兼容模式下 op-plugin 传
            # USE_FP32_ADD(4)，golden 同步走 fp32 域融合。
            cube_math_type = _compat_impl_cmt(x1, x2, x3, fused_op_type)
        return torch_fused_matmul_core(
            x1,
            x2,
            bias,
            x3,
            fused_op_type=fused_op_type,
            out_dtype=out_dtype,
            cube_math_type=cube_math_type,
        )


class CannOpsNnFusedMatmulTestSpec:
    @staticmethod
    def golden(
        x,
        x2,
        *,
        bias=None,
        x3=None,
        alpha=None,
        beta=None,
        fused_op_type="",
        **kwargs,
    ):
        """torch.ops.cann_ops_nn.fused_matmul: y = alpha*(x@x2) + beta*x3 (via aclnnFusedMatmulV2)."""
        out_dtype = torch.float32 if fused_op_type == "16cast32" else x.dtype
        alpha_value = 1.0 if alpha is None else float(alpha)
        beta_value = 1.0 if beta is None else float(beta)
        return torch_fused_matmul_core(
            x,
            x2,
            bias,
            x3,
            fused_op_type=fused_op_type,
            out_dtype=out_dtype,
            cube_math_type=_cann_ops_cmt(x, x2, x3, fused_op_type),
            alpha=alpha_value,
            beta=beta_value,
        )


__spec__ = {
    "torch_npu.npu_fused_matmul": "TorchNpuFusedMatmulTestSpec",
    "torch.ops.cann_ops_nn.fused_matmul": "CannOpsNnFusedMatmulTestSpec",
}
