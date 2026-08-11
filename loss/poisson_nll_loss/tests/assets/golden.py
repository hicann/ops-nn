#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
TTK TestSpec for poisson_nll_loss (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— CPU 真值，竞品接口 F.poisson_nll_loss（非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一竞品接口在远端 GPU 上执行（cross_check 比值的另一腿）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

Canonical IO order (poisson_nll_loss_def.cpp):
    inputs : input_x, target
    outputs: loss
    attrs  : log_input(OPTIONAL bool=True), full(OPTIONAL bool=False),
             eps(OPTIONAL float=1e-8), reduction(OPTIONAL str="mean")

kernel 内部 fp16->fp32 计算再舍回，golden 同样先抬 fp32 计算、最后照 output_dtypes 舍回
（中间精度与算子实现一致，不抬 fp64）。
"""

import numpy as np
import torch
import torch.nn.functional as F

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """attributes 可能直接平铺在 kwargs，也可能收在 kwargs['attributes'] dict；字符串做类型归一。"""
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "false", "yes", "no", "1", "0"):
            return s in ("true", "yes", "1")
        try:
            return type(default)(v)
        except Exception:
            return default
    return v


def _f32_floor(t):
    """fp16/bf16 抬 fp32（CPU half 支持残缺，且与 NPU 内核 fp32 计算一致）；
    fp32/fp64 照单全收——cross_check 场景框架按 golden_mode=Promote 自动把输入抬一档
    （fp32→fp64），golden 不自行 cast、不替框架做精度决策（砍回 fp32 会废掉 Promote）。
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _compute(input_x, target, **kwargs):
    """torch.Tensor 进 / 出，返回 list[Tensor]，顺序照 def.cpp 输出序（仅 loss）。"""
    log_input = bool(_attr(kwargs, "log_input", True))
    full = bool(_attr(kwargs, "full", False))
    eps = float(_attr(kwargs, "eps", 1e-8))
    reduction = str(_attr(kwargs, "reduction", "mean"))
    y = F.poisson_nll_loss(
        _f32_floor(input_x),
        _f32_floor(target),
        log_input=log_input,
        full=full,
        eps=eps,
        reduction=reduction,
    )
    return [y]


class _PoissonNllLossCompose:
    """三方标杆：竞品接口 F.poisson_nll_loss 直出，由 TTK 派发到远端 GPU 执行。

    参数绑定契约：属性喂 __init__、输入喂 __call__，参数名与 def.cpp 逐字一致
    （input_x / target）。输出 dtype 天然随输入（= NPU 输出 dtype），无需额外 cast。
    """

    def __init__(self, *, log_input=True, full=False, eps=1e-8, reduction="mean", **_):
        self.log_input = bool(log_input)
        self.full = bool(full)
        self.eps = float(eps)
        self.reduction = str(reduction)

    def __call__(self, input_x, target, **_):
        return [
            F.poisson_nll_loss(
                input_x,
                target,
                log_input=self.log_input,
                full=self.full,
                eps=self.eps,
                reduction=self.reduction,
            )
        ]


class PoissonNllLossSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray]。"""

    def golden(input_x, target, **kwargs):
        inp = torch.from_numpy(np.ascontiguousarray(input_x))
        tgt = torch.from_numpy(np.ascontiguousarray(target))
        outs = _compute(inp, tgt, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy().astype(input_x.dtype)
            for i, o in enumerate(outs)
        ]

    third_party = {"torch": _PoissonNllLossCompose}
    tolerance = _TOL


def poisson_nll_loss_golden(input_x, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return PoissonNllLossSpec.golden(input_x, target, **kwargs)[0]


__spec__ = {"poisson_nll_loss": "PoissonNllLossSpec"}
__golden__ = {"kernel": {"poisson_nll_loss": "poisson_nll_loss_golden"}}

# 【不存在】aclnn 通路：A2 侧无 aclnn 接口（def ACLNNTYPE aclnn_exclude，对齐 A2 不新增），
#   见 01_requirement.md §3.2。
# 【不存在】e2e 通路：torch poisson_nll_loss 是 composite 分解，不走本融合算子；
#   torch_npu 二进制无 aclnnPoissonNllLoss 引用（dispatcher 权威判定，01 §3.2）。
# 【不存在】tf 通路：A2 canndev 无 tf_plugin 本体。
