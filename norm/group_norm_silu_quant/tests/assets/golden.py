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

    # 可选输入缺省:gamma/beta 是 OPTIONAL 参数, 不传时按算子语义取 gamma=1、beta=0
    # (kernel arch35 ..._split_reduce.h:265-266 `hasGamma ? gF[ci] : 1.0f` / `hasBeta ? bF[ci] : 0.0f`)。
    # 此前直接 _f32(gamma) 会在 gamma 为 None 时抛异常, 导致该档用例判成 GOLDEN_FAILURE ——
    # 等于"可选输入不给"这一档从来没被真正验证过(issue #21 的 L1_014 正是这一档)。
    if gamma is None:
        gamma = np.ones((C,), dtype=np.float32)
    if beta is None:
        beta = np.zeros((C,), dtype=np.float32)

    # 空 Tensor(任意维度为 0):N==0 时下面的 reshape/var_mean 会抛异常(判成 GOLDEN_FAILURE), 需短路。
    # 空 Tensor 契约:**meanOut 填 0, rstdOut 填 NAN**。
    # 权威依据是本算子自己的两处资料(README.md:73 / docs/aclnnGroupNormSiluQuant.md:112)与手写 aclnn
    # 的实现(op_host/op_api/aclnn_group_norm_silu_quant.cpp:251 `IsEmpty()` 分支显式
    # FillScalar(meanOut, 0) + FillScalar(rstdOut, NAN) 后直接返回, 不下发内核)。
    # ⚠️ arch35 内核的 empty 分支原本对 mean 也填 NAN, 与 aclnn 通路语义不一致(GE 图通路才会走到),
    # 已在 op_kernel/arch35/..._empty_tensor.h 修正为填 0。golden 对齐契约, 不对齐一时的实现。
    if xf.size == 0:
        y_empty = np.zeros(np.asarray(x).shape, dtype=np.int8)
        mean_e = np.zeros((N, G), dtype=np.float32)
        rstd_e = np.full((N, G), np.nan, dtype=np.float32)
        return [y_empty, _cast_back(mean_e, mean_dt), _cast_back(rstd_e, rstd_dt)]

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


# 模块级别名:类体内直接引用 `__golden_...` 会触发 Python 的 name mangling
# (被改写成 _GroupNormSiluQuantSpec__golden_...), 导致 Spec.golden 调用时 NameError。
_golden_impl = __golden_group_norm_silu_quant


class GroupNormSiluQuantSpec:
    """判据声明。**必须显式给 int8 输出声明 quant 标准**:
    y 是量化输出, 不声明时 TTK 会把 int8 硬路由到 binary_equal(逐位相等), 量化舍入产生的 ±1 LSB
    会被大面积误判为失败——实测新配额集 211 例里 26 例"失败"全部是 |diff|==1(最小那例 dump:
    32 个元素中 6 个差 1、最大 |diff|=1)。quant 标准的判定是 |out-golden|>1 才计错, 且 ptol 默认 0
    (一个都不许超), 既不放水也不误杀。

    浮点输出(mean/rstd)声明 CANN 开源精度标准 `stat_rel_err`(mere < th 且 mare < 10*th, th 按 dtype 取
    2^-8/2^-10/2^-13)。**不能沿用 TTK 默认的 isclose**:其 atol=1e-8 低于 fp32/bf16 在常见量级上的分辨率
    (bf16 尾数仅 8 位, 1 ULP 的相对误差就有 ~0.4%), 等价于要求逐位相等。实测 case00603_rg(2,5120,7) bf16:
    2560 个 mean 元素里 10 个不相等, **每一个都恰好差 1 ULP**(ULP 倍数 max=1.0000, 超 1 ULP 的 0 个)
    —— 内核 fp32 累加后舍入到 bf16 与 torch 求和次序不同, 边界值差一格, 任何实现都做不到更好。
    """

    tolerance = {
        "int8": {"standard": "quant"},
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    @staticmethod
    def golden(x, gamma, beta, quant_scale, **kwargs):
        return _golden_impl(x, gamma, beta, quant_scale, **kwargs)


__spec__ = {"group_norm_silu_quant": "GroupNormSiluQuantSpec"}
__golden__ = {"kernel": {"group_norm_silu_quant": "__golden_group_norm_silu_quant"}}
