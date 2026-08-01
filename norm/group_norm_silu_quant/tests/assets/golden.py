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
"""
TTK golden plugin for group_norm_silu_quant (kernel mode, arch35/Ascend950).

**Implemented with torch tensor ops** (torch.var_mean / rsqrt / addcmul / F.silu / round+clamp),
not hand-written numpy formulas.

刻意不用层级 API `F.group_norm`：它带 BN 系列的训练态保护
（"Expected more than 1 value per channel when training"），每通道只剩 1 个元素时直接拒收
（2 维输入 (N,C) 且 group=C 即触发，实测 case00608 崩）。那是网络层的约束，与本算子的
数学定义无关——算子本身 elemNum=1 合法。张量算子拼接同样满足红线 R3「竞品算子拼接实现」。 红线 R3：golden 只能"竞品接口实现"或"竞品算子拼接实现"，禁 numpy 纯公式——
纯公式与被测内核容易犯同一个错，拿和内核一样烂的参照去比，会把精度短板伪装成达标。

Reference for IO/attr semantics:
    ops-nn/norm/group_norm_silu_quant/tests/st/aclnnGroupNormSiluQuant/executor_aclnnGroupNormSiluQuant.py
and the requirement/design docs (changwei_deliverables/GroupNormSiluQuant/).

Compute (per group of elemNum = (C/num_groups)*HW elements):
    mean = mean(x)                          # over each group
    var  = var(x, ddof=0)                   # population variance
    rstd = 1/sqrt(var + eps)
    x_norm = (x - mean) * rstd
    y_f    = x_norm * gamma[c] + beta[c]     # per-channel affine
    silu   = activate_silu ? y_f*sigmoid(y_f) : y_f
    y_i8   = clamp(round(silu / quantScale), -128, 127)   # per-tensor(len 1) or per-channel(len C)
    meanOut/rstdOut: per group, shape (N, num_groups), cast back to x dtype.

Canonical IO order (group_norm_silu_quant_def.cpp):
    inputs : x, gamma, beta, quantScale      (x/gamma/beta same dtype bf16|fp16; quantScale fp32)
    outputs: yOut(int8), meanOut, rstdOut
    attrs  : num_groups(REQUIRED int), eps(OPTIONAL float=1e-5), activate_silu(OPTIONAL bool=True)

TTK passes input arrays positionally in CSV input_shapes order; attributes + output_dtypes via **kwargs.
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def _f32(a):
    return np.asarray(a).astype(np.float32)


def _cast_back(arr_f32, target):
    if target == "bfloat16":
        return (
            arr_f32.astype(_bf16) if _bf16 is not None else arr_f32.astype(np.float32)
        )
    return arr_f32.astype(target)


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def __golden_group_norm_silu_quant(x, gamma, beta, quant_scale, **kwargs):
    num_groups = int(_attr(kwargs, "num_groups", 1))
    eps = torch.from_numpy(np.asarray([_attr(kwargs, "eps", 1e-5)], dtype=np.float32))[
        0
    ]
    silu = _attr(kwargs, "activate_silu", True)
    if isinstance(silu, str):
        silu = silu.strip().lower() in ("1", "true", "yes")
    silu = bool(silu)

    output_dtypes = kwargs.get("output_dtypes")

    def _od(i, default):
        if output_dtypes and i < len(output_dtypes):
            od = output_dtypes[i]
            return od[0] if isinstance(od, (tuple, list)) else str(od)
        return default

    x_dt = str(np.asarray(x).dtype)
    mean_dt = _od(1, x_dt)
    rstd_dt = _od(2, x_dt)

    xf = _f32(x)
    N, C = xf.shape[0], xf.shape[1]
    HW = int(np.prod(xf.shape[2:])) if xf.ndim > 2 else 1
    G = num_groups

    # ── 用 torch 库算子拼接，不手写公式 ──
    # torch.var_mean(unbiased=False) 给出与算子定义一致的总体方差与均值。
    xt = torch.from_numpy(xf).reshape(N, C, HW)
    gt = torch.from_numpy(_f32(gamma).reshape(-1))
    bt = torch.from_numpy(_f32(beta).reshape(-1))

    xg = xt.reshape(N, G, -1)
    var_t, mean_t = torch.var_mean(xg, dim=-1, unbiased=False, keepdim=True)
    rstd_t = torch.rsqrt(var_t + eps)

    # ⚠️ 不能直接用 F.group_norm：每通道只剩 1 个元素时（如 2 维输入 (N,C) 且 group=C）
    # 它会抛 "Expected more than 1 value per channel when training"——那是 BN 系列的
    # 训练态保护，对本算子不适用（算子本身支持 elemNum=1）。实测 case00608 ((1,24),...) 即崩。
    # 改用 torch 的归一化+仿射算子拼接，语义与 F.group_norm 一致但不带这条限制：
    #   normalized = (x - mean) * rstd   （mean/rstd 已由 torch.var_mean 给出，同一套统计量）
    #   out = normalized * gamma[c] + beta[c]
    gn_t = ((xg - mean_t) * rstd_t).reshape(N, C, HW)
    out_t = torch.addcmul(bt.reshape(1, C, 1), gn_t, gt.reshape(1, C, 1))
    if silu:
        out_t = F.silu(out_t)

    qs = _f32(quant_scale).reshape(-1)
    if qs.size == C:
        scale_t = torch.from_numpy(qs).reshape(1, C, 1)  # per-channel
    else:
        scale_t = torch.tensor(float(qs[0]), dtype=torch.float32)  # per-tensor
    q_t = torch.clamp(torch.round(out_t / scale_t), -128.0, 127.0)
    y_i8 = q_t.to(torch.int8).numpy().reshape(np.asarray(x).shape)

    mean = mean_t.numpy()
    rstd = rstd_t.numpy()

    mean_out = _cast_back(mean.reshape(N, G), mean_dt)
    rstd_out = _cast_back(rstd.reshape(N, G), rstd_dt)
    return [y_i8, mean_out, rstd_out]


__golden__ = {"kernel": {"group_norm_silu_quant": "__golden_group_norm_silu_quant"}}
