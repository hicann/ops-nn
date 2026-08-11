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
TTK TestSpec for l2_normalize_grad (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（sum/sqrt/clamp/逐点乘除），fp64 计算（理由见下）
    third_party  —— 三方标杆，torch 拼接在远端 GPU 上执行（fp32，竞品自然精度）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

公式源自 ascend910b 的算法规格（l2_normalize_grad.py），不是被测内核。与内核对齐的只有
*输入契约*：两者都直接消费传入的 y、而非从 x 重算，这样 (x, y, dy) 不自洽的随机三元组
两边处理一致——否则每条随机用例都会变成假失败（与 rms_norm_grad golden 消费传入 rstd 同理）：
    n  = max(sqrt(sum(x*x, dim)), eps)
    s  = sum(y*dy, dim)
    dx = (dy - y*s) / n
当 y == F.normalize(x, p=2, dim, eps)（一致输入、||x|| > eps 的正常量级）时，等价于
torch autograd 经 F.normalize 的反向（见 00_spec 2 / 6.1）。

为什么 golden 必须 fp64（特事特办，非一般规则）：dx = dy - y*s 是对消差，
fp32 golden 自带规约/对消误差、可能误flag内核；golden 取 fp64 真值（torch 拼接，
非 numpy 纯公式，红线 R3），与 instance/in_training 的 fp64 golden 约定一致。

Canonical IO order (l2_normalize_grad_def.cpp):
    inputs : x, y, dy（同 dtype）
    outputs: dx（同 x dtype/shape）
    attrs  : dim(OPTIONAL ListInt={1}), eps(OPTIONAL float=1e-4)
"""

import numpy as np
import torch

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _resolve_axis(dim, rank):
    if dim is None:
        return 1
    if isinstance(dim, (list, tuple, np.ndarray)):
        vals = list(dim)
        axis = int(vals[0]) if len(vals) > 0 else 1
    else:
        axis = int(dim)
    if axis < 0:
        axis += rank
    return axis


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _compute(x, y, dy, **kwargs):
    """torch.Tensor 进 / 出（fp64 真值），返回 [dx]，顺序照 def.cpp。"""
    axis = _resolve_axis(_attr(kwargs, "dim", (1,)), x.dim())
    eps = float(_attr(kwargs, "eps", 1e-4))

    xf = x.to(torch.float64)
    yf = y.to(torch.float64)
    dyf = dy.to(torch.float64)

    # ── 以下全部为 torch 库算子拼接，不手写 numpy 数值公式（红线 R3）──
    sq = (xf * xf).sum(dim=axis, keepdim=True)
    n = torch.clamp(torch.sqrt(sq), min=eps)
    s = (yf * dyf).sum(dim=axis, keepdim=True)
    dx = (dyf - yf * s) / n
    return [dx]


class _L2NormalizeGradCompose:
    """三方标杆：torch 拼接在远端 GPU 执行，fp32（竞品自然精度，不抬 fp64——
    否则分母趋零、cross_check 比值爆表，会把内核误判成缺陷；三方须同精度对等）。
    参数名与 def.cpp 逐字一致（x/y/dy/dim/eps）。输出与 NPU 同 dtype，无需额外 cast。
    """

    def __init__(self, *, dim=(1,), eps=1e-4, **_):
        self.dim = dim
        self.eps = float(eps)

    def __call__(self, x, y, dy, **_):
        axis = _resolve_axis(self.dim, x.dim())
        sq = (x * x).sum(dim=axis, keepdim=True)
        n = torch.clamp(torch.sqrt(sq), min=self.eps)
        s = (y * dy).sum(dim=axis, keepdim=True)
        return [(dy - y * s) / n]


class L2NormalizeGradSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray](fp64 真值)。"""

    def golden(x, y, dy, **kwargs):
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(dy)),
            **kwargs,
        )
        return [o.numpy().astype(np.float64) for o in outs]

    third_party = {"torch": _L2NormalizeGradCompose}
    tolerance = _TOL


def l2_normalize_grad_golden(x, y, dy, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return L2NormalizeGradSpec.golden(x, y, dy, **kwargs)[0]


__spec__ = {"l2_normalize_grad": "L2NormalizeGradSpec"}
__golden__ = {"kernel": {"l2_normalize_grad": "l2_normalize_grad_golden"}}

# 【不存在】aclnn 通路：canndev/ops-nn 均无 op_api/l2_normalize_grad（01 §3.3，GE 梯度图专用反向算子）。
# 【不存在】e2e 通路：torch_npu 二进制无 aclnnL2NormalizeGrad 引用（全库无 L2Normalize 串，01 §3.3）。
# 【不存在】tf / onnx / caffe 通路：framework 插件无 L2NormalizeGrad 注册（01 §3.3）。
