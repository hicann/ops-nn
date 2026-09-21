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

"""mse_loss_grad_v2 多通路 golden（TestSpec 范式，骨架照 changwei-op-dev templates/golden_spec.template.py）。

迁移自 test6/op（独立 ascend950 实现）→ ops-nn loss/mse_loss_grad_v2。通路支持表：

| 通路   | 支持 | 依据 |
|--------|------|------|
| kernel | ✅   | op_kernel/arch35/mse_loss_grad_v2.cpp（TilingKey 分 RANK4/8） |
| geir   | ✅   | 内置 proto（vendor op_proto/inc/mse_loss_grad_v2_proto.h REG_OP(MseLossGradV2)）+ infershape |
| aclnn  | ✅   | op_host/op_api/aclnn_mse_loss_backward.{h,cpp} + docs/aclnnMseLossBackward.md |
| e2e    | ❌   | 交付包不含 torch 绑定（torch 接口属 torch_npu 仓绑定，未交付即无 e2e 通路） |
| tf     | ❌   | 交付包无 tf plugin .so |

算法语义（按算子契约写，不照抄内核实现）：

    cof = 2 / numel(predict)  (reduction="mean")   或   2.0  (none/sum)
    y   = (predict - label) * cof * dout           （三输入 numpy broadcast）

golden 用竞品接口实现（红线 R3）：torch.ops.aten.mse_loss_backward（PyTorch 官方 MSE
反向单一融合算子，aten 位置形态 (grad_output, self, target, reduction:int)）。
"""

__spec__ = {
    # ── kernel + geir：同一注册键（OpDef 蛇形名），共用一个类 ──
    "mse_loss_grad_v2": "MseLossGradV2KernelSpec",
    # ── aclnn：键 = aclnn 接口名（ops-nn 既有） ──
    "aclnnMseLossBackward": "MseLossGradV2AclnnSpec",
    # ── e2e（torch 单算子）：键 = torch API 全路径（aten::mse_loss_backward） ──
    "torch.ops.aten.mse_loss_backward": "MseLossGradV2E2ESpec",
}

import numpy as np
import torch

try:
    import ml_dtypes

    _BF16 = np.dtype(ml_dtypes.bfloat16)
except ImportError:  # pragma: no cover
    _BF16 = None

# ── 判据 tolerance ──
# kernel/GEIR：浮点输出 cross_check L1（golden 自动 Promote：fp16/bf16→fp32、fp32→fp64）。
_TOL = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}

# aclnn 通路三方免检 → 本机判（红线：TTK aclnn 三方 party down 会全假失败）。
_TOL_LOCAL = {
    "float32": {"standard": "mix_tolerance"},
    "float16": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
}

_REDUCTION_STR_TO_CODE = {"none": 0, "mean": 1, "sum": 2}


def _reduction_code(reduction):
    """reduction 归一化：kernel/GEIR 字符串形态与 aclnn int64 形态都收。

    非法取值按 spec error_codes.attribute_value_out_of_range 语义直接拒绝。
    """
    if reduction is None:
        return 1  # OPTIONAL 缺省 = "mean"
    if isinstance(reduction, str):
        key = reduction.strip().lower()
        if key not in _REDUCTION_STR_TO_CODE:
            raise ValueError(
                f"reduction must be one of none/mean/sum, got {reduction!r}"
            )
        return _REDUCTION_STR_TO_CODE[key]
    code = int(reduction)
    if code not in (0, 1, 2):
        raise ValueError(f"reduction (int64 form) must be in [0, 2], got {reduction!r}")
    return code


# ══════════════════════════════════════════════════════════════════════
# 计算核 —— 全算子只写这一份，两条通路共用（竞品接口实现，R3）
# ══════════════════════════════════════════════════════════════════════
def _compute(predict, label, dout, reduction):
    """全程 torch.Tensor 进出。y = (predict - label) * cof * dout（broadcast）。

    精度契约（Promote）：cross_check 场景框架把输入抬一档喂进来（fp16/bf16→fp32、
    fp32→fp64），此处照单全收、绝不向下砍；aten.mse_loss_backward 原生支持
    fp32/fp64 与 fp16/bf16，无需任何 cast。
    空 tensor（numel=0）输出为空；cof 的 2/N 由 aten 内部按 reduction 处理。
    """
    code = _reduction_code(reduction)
    # 返回 list[Tensor]：裸 Tensor 会被调用方按首维迭代
    return [torch.ops.aten.mse_loss_backward(dout, predict, label, code)]


def _np_to_torch(a):
    """numpy（含 ml_dtypes bfloat16）→ torch.Tensor，按位无损。"""
    a = np.ascontiguousarray(a)
    if _BF16 is not None and a.dtype == _BF16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


def _torch_to_np(t):
    """torch.Tensor → numpy，bf16 按位回投 ml_dtypes.bfloat16。"""
    if t.dtype == torch.bfloat16:
        u16 = t.contiguous().view(torch.uint16).cpu().numpy()
        return u16.view(_BF16) if _BF16 is not None else u16
    return t.cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# third_party —— 三方精度 + 三方性能共用（竞品 = PyTorch 官方反向算子，
# 即竞品最优形态：单一 fused 算子而非 Sub/Mul 分解拼接）
# ══════════════════════════════════════════════════════════════════════
class _TorchRef:
    """XPU server 按名绑定：属性喂 __init__、输入喂 __call__。

    输入名 = proto REG_OP 注册名（predict/label/dout）；竞品输出 dtype 跟随输入
    （= NPU 输出 dtype），天然同精度对等；防御性再 cast 一次 predict.dtype。
    """

    def __init__(self, reduction="mean", **kwargs):
        self.reduction_code = _reduction_code(reduction)

    def __call__(self, predict, label, dout, **kwargs):
        out = torch.ops.aten.mse_loss_backward(
            dout, predict, label, self.reduction_code
        )
        return [out.to(predict.dtype)]


# ══════════════════════════════════════════════════════════════════════
# 各通路的壳 —— 只做容器转换，不写计算逻辑
# ══════════════════════════════════════════════════════════════════════
class MseLossGradV2KernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray，返 numpy.ndarray。

    参数名取自 op_host/mse_loss_grad_v2_def.cpp：输入 predict/label/dout（按位），
    属性 reduction（关键字）。
    """

    def golden(predict, label, dout, reduction=None, **kwargs):
        outs = _compute(
            _np_to_torch(predict), _np_to_torch(label), _np_to_torch(dout), reduction
        )
        return [_torch_to_np(t) for t in outs]

    third_party = {"torch": _TorchRef}
    tolerance = _TOL


class MseLossGradV2AclnnSpec:
    """aclnn 通路。golden 收 torch.Tensor，返 torch.Tensor。

    ★位置参数顺序必须与 aclnn C header 逐字一致（TTK AclnnParamPlan.build_args 按
    header param_layout 顺序喂位置参数）：
      aclnnMseLossBackwardGetWorkspaceSize(gradOutput, self, target, int64_t reduction, out, ...)
    其中 reduction 为 int64 编码 0('none')/1('mean')/2('sum')。
    映射：gradOutput=dout、self=predict、target=label。
    """

    def golden(gradOutput, self, target, reduction=None, out=None, **kwargs):
        t = [
            x.cpu() if isinstance(x, torch.Tensor) else x
            for x in (gradOutput, self, target)
        ]
        # t = [dout, predict, label]
        outs = _compute(t[1], t[2], t[0], reduction)
        return outs

    # 【预留】aclnn 通路三方：TTK 版本支持即自动生效，不生效也无副作用。
    third_party = {"torch": _TorchRef}
    tolerance = _TOL_LOCAL


class MseLossGradV2E2ESpec:
    """e2e（torch 单算子）通路。golden 收 torch.Tensor，返 torch.Tensor。

    位置参数顺序 = aten schema：`aten::mse_loss_backward(grad_output, self, target,
    int reduction) -> Tensor`；映射 grad_output=dout、self=predict、target=label。
    """

    def golden(grad_output, self, target, reduction=None, **kwargs):
        outs = _compute(self, target, grad_output, reduction)
        return outs

    tolerance = _TOL_LOCAL
