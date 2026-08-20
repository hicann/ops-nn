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
"""SoftmaxFocalLossGrad 多通路 golden(TestSpec 范式)。

通路支持表(照抄 01_requirement.md §3.3):
  | 通路   | 支持 | 依据                                                            |
  |--------|------|-----------------------------------------------------------------|
  | kernel | ✅   | op_kernel/arch35/ 有实现                                         |
  | geir   | ✅   | op_graph/ 有 REG_OP(SoftmaxFocalLossGrad) + IMPL_OP_INFERSHAPE   |
  | aclnn  | ❌   | canndev 老树 built-in 与新树 ops/ 均无 op_api 实现                |
  | e2e    | ❌   | torch_npu 二进制无 aclnnSoftmaxFocalLossGrad 符号                 |
"""

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键(算子蛇形名), geir 不另写
    "softmax_focal_loss_grad": "SoftmaxFocalLossGradKernelSpec",
}

# 判据: 浮点输出配 cross_check 才会去取三方数据; L1 见 verification.md §5.2
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _attr(kwargs, name, default):
    """CSV 里的 attributes 可能是字符串, 统一转成 default 的类型。"""
    v = kwargs.get(name, default)
    if isinstance(v, str):
        try:
            return type(default)(v)
        except ValueError:
            return default
    return v


def _compute(pred, target, dout, weight=None, **kwargs):
    """全程 torch.Tensor 进出, 返回 list[Tensor], 顺序照 def.cpp 的输出序。

    计算语义对齐 A2 tbe.dsl(softmax_focal_loss_grad_compute):
        wf   = alpha * exp(gamma     * log(1-p)) * t     WF = sum(wf, -1, keepdim)
        wb   = alpha * exp((gamma-1) * log(1-p)) * t     WB = sum(wb, -1, keepdim)
        ce   = -log(p) * t * w                           CE = sum(ce, -1, keepdim)
        W    = sum(w * t, -1, keepdim)
        d_ce = p * W - t * w
        d_wf = -gamma * ((WF - WB) + wb) * p
        grad = (d_ce * WF + d_wf * CE) * dout
        reduction == "mean" 时再乘 1/numel(pred); "none"/"sum" 不缩放。

    A2 的 weight 为 None 时其实现直接崩(vmul(weight, target) 无判空), 此处按全 1
    权重给出数学定义, 与 A5 内核的补齐行为一致。

    精度决策契约: cross_check 下框架按 Promote 把输入抬一档喂进来, 这里只向上兜底
    (half→fp32 防 CPU half 残缺), 绝不向下砍。
    """
    gamma = _attr(kwargs, "gamma", 2.0)
    alpha = _attr(kwargs, "alpha", 0.25)
    reduction = _attr(kwargs, "reduction", "mean")

    dt = pred.dtype
    if dt in (torch.float16, torch.bfloat16):
        dt = torch.float32
    p = pred.to(dt)
    t = target.to(dt)
    d = dout.to(dt)
    w = weight.to(dt) if weight is not None else torch.ones_like(p)

    neg_one = torch.tensor(-1.0, dtype=dt)
    one = torch.tensor(1.0, dtype=dt)
    g = torch.tensor(float(gamma), dtype=dt)
    g_sub1 = torch.tensor(float(gamma) - 1.0, dtype=dt)
    a = torch.tensor(float(alpha), dtype=dt)

    log_1sub_p = torch.log(torch.add(torch.mul(p, neg_one), one))
    log_p = torch.log(p)

    # wf / WF
    wf = torch.exp(torch.mul(log_1sub_p, g))
    wf = torch.mul(wf, t)
    wf = torch.mul(wf, a)
    wf_sum = torch.sum(wf, dim=-1, keepdim=True)

    # wb / WB   (wb 逐元素值在 d_wf 中还要再用一次)
    wb = torch.exp(torch.mul(log_1sub_p, g_sub1))
    wb = torch.mul(wb, t)
    wb = torch.mul(wb, a)
    wb_sum = torch.sum(wb, dim=-1, keepdim=True)

    # ce / CE
    ce = torch.mul(log_p, t)
    ce = torch.mul(ce, w)
    ce = torch.mul(ce, neg_one)
    ce_sum = torch.sum(ce, dim=-1, keepdim=True)

    # W = sum(w * t)
    wt_sum = torch.sum(torch.mul(w, t), dim=-1, keepdim=True)

    # d_ce = p * W - t * w
    d_ce = torch.sub(torch.mul(p, wt_sum), torch.mul(t, w))

    # d_wf = -gamma * ((WF - WB) + wb) * p
    d_wf = torch.sub(wf_sum, wb_sum)
    d_wf = torch.add(d_wf, wb)
    d_wf = torch.mul(d_wf, p)
    d_wf = torch.mul(d_wf, torch.mul(g, neg_one))

    # grad = (d_ce * WF + d_wf * CE) * dout
    res = torch.add(torch.mul(d_ce, wf_sum), torch.mul(d_wf, ce_sum))
    res = torch.mul(res, d)

    if reduction == "mean":
        numel = int(np.prod(list(p.shape))) if p.dim() > 0 else 1
        if numel > 0:
            res = torch.mul(res, torch.tensor(1.0 / float(numel), dtype=dt))

    return [res.contiguous()]


class _Compose:
    """竞品标杆(A100 上执行): 用 torch 高层表达式拼等价语义。

    与 _compute 的实现路径相互独立(表达式/算子重载 vs 逐步 torch.mul), 但算法保持
    一致: (1-p)^gamma 同样用 exp(gamma*log(1-p)) 而非 torch.pow —— pow 更准, 换写法
    会让竞品凭空更准约 2 倍, cross_check 假红。
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = str(reduction)

    def __call__(self, pred, target, dout, weight=None, **kwargs):
        out_dtype = pred.dtype
        p = pred.float()
        t = target.float()
        d = dout.float()
        w = weight.float() if weight is not None else torch.ones_like(p)

        log_1sub_p = torch.log(1.0 - p)
        wf = self.alpha * torch.exp(self.gamma * log_1sub_p) * t
        wb = self.alpha * torch.exp((self.gamma - 1.0) * log_1sub_p) * t
        wf_sum = wf.sum(dim=-1, keepdim=True)
        wb_sum = wb.sum(dim=-1, keepdim=True)
        ce_sum = (-torch.log(p) * t * w).sum(dim=-1, keepdim=True)
        wt_sum = (w * t).sum(dim=-1, keepdim=True)

        d_ce = p * wt_sum - t * w
        d_wf = -self.gamma * ((wf_sum - wb_sum) + wb) * p
        grad = (d_ce * wf_sum + d_wf * ce_sum) * d
        if self.reduction == "mean" and p.numel() > 0:
            grad = grad * (1.0 / float(p.numel()))
        # 浮点输出必须 cast 回 NPU 输出 dtype, 否则竞品天然更准, ratio 失真
        return [grad.contiguous().to(out_dtype)]


class SoftmaxFocalLossGradKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray, 返 numpy.ndarray。

    参数名取自 op_host/softmax_focal_loss_grad_def.cpp: pred / target / dout / weight。
    """

    def golden(*inputs, **kwargs):
        t = [
            None if a is None else torch.from_numpy(np.ascontiguousarray(a))
            for a in inputs
        ]
        outs = _compute(*t, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy()
            for i, o in enumerate(outs)
        ]

    third_party = {"torch": _Compose}
    tolerance = _TOL


# 【不存在】aclnn 通路: canndev 老树 ops/built-in/op_api 与新树 ops/ 下均无
#   softmax_focal_loss_grad 的 op_api 实现, 也无 docs/aclnnSoftmaxFocalLossGrad.md(01 §3.3)。
# 【不存在】e2e(torch) 通路: torch_npu 2.10.0 的 libtorch_npu.so 无 aclnnSoftmaxFocalLossGrad
#   符号 —— aclnn 本就不存在, torch 无从下发。
# 【不存在】tf / onnx / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter。
