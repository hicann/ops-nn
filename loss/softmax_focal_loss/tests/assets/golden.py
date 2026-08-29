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
"""SoftmaxFocalLoss 多通路 golden(TestSpec 范式)。

通路支持表(照抄 01_requirement.md §3.3):
  | 通路   | 支持 | 依据                                                        |
  |--------|------|-------------------------------------------------------------|
  | kernel | ✅   | op_kernel/arch35/ 有实现                                     |
  | geir   | ✅   | op_graph/ 有 REG_OP(SoftmaxFocalLoss) + IMPL_OP_INFERSHAPE   |
  | aclnn  | ❌   | canndev 老树 built-in 与新树 ops/ 均无 op_api 实现            |
  | e2e    | ❌   | torch_npu 二进制无 aclnnSoftmaxFocalLoss 符号                 |
"""

import numpy as np
import torch

__spec__ = {
    # kernel + geir 共用同一个注册键(算子蛇形名), geir 不另写
    "softmax_focal_loss": "SoftmaxFocalLossKernelSpec",
}

# 判据: 浮点输出配 cross_check 才会去取三方数据; L1 见 verification.md §5.2


def _attr(kwargs, name, default):
    """CSV 里的 attributes 可能是字符串, 统一转成 default 的类型。"""
    v = kwargs.get(name, default)
    if isinstance(v, str):
        try:
            return type(default)(v)
        except ValueError:
            return default
    return v


def _compute(pred, target, weight=None, **kwargs):
    """全程 torch.Tensor 进出, 返回 list[Tensor], 顺序照 def.cpp 的输出序。

    计算语义对齐 A2 tbe.dsl(softmax_focal_loss_compute):
        ce = -log(pred) * target * weight        CE = sum(ce, -1, keepdim)
        fw = alpha * exp(gamma * log(1 - pred)) * target
                                                 FW = sum(fw, -1, keepdim)
        y  = broadcast(CE * FW)                  # 整行同值
    reduction 在计算侧不生效(与 A2 同): 输出恒与 pred 同形。

    weight 缺省时按全 1 权重给出数学定义, 与 A2 实现(逐处判空后跳过加权)及 A5 内核
    的补齐行为一致。

    精度决策契约: cross_check 下框架按 Promote 把输入抬一档喂进来, 这里只向上兜底
    (half→fp32 防 CPU half 残缺), 绝不向下砍。
    """
    gamma = _attr(kwargs, "gamma", 2.0)
    alpha = _attr(kwargs, "alpha", 0.25)

    dt = pred.dtype
    if dt in (torch.float16, torch.bfloat16):
        dt = torch.float32
    p = pred.to(dt)
    t = target.to(dt)
    w = weight.to(dt) if weight is not None else torch.ones_like(p)

    neg_one = torch.tensor(-1.0, dtype=dt)
    one = torch.tensor(1.0, dtype=dt)
    g = torch.tensor(float(gamma), dtype=dt)
    a = torch.tensor(float(alpha), dtype=dt)

    # ce = -log(p) * t * w  →  CE
    ce = torch.mul(torch.log(p), neg_one)
    ce = torch.mul(ce, t)
    ce = torch.mul(ce, w)
    ce_sum = torch.sum(ce, dim=-1, keepdim=True)

    # fw = alpha * exp(gamma * log(1 - p)) * t  →  FW
    p_1sub = torch.add(torch.mul(p, neg_one), one)
    fw = torch.exp(torch.mul(torch.log(p_1sub), g))
    fw = torch.mul(fw, t)
    fw = torch.mul(fw, a)
    fw_sum = torch.sum(fw, dim=-1, keepdim=True)

    # y = CE * FW, 广播回整行
    res = torch.mul(ce_sum, fw_sum).expand_as(p).contiguous()
    return [res]


class _Compose:
    """竞品标杆(A100 上执行): 用 torch 高层表达式拼等价语义。

    与 _compute 的实现路径相互独立(这里走表达式/算子重载, _compute 走逐步 torch.mul),
    但算法保持一致: (1-p)^gamma 同样用 exp(gamma*log(1-p)) 而非 torch.pow ——
    pow 比 exp∘log 更准, 换写法会让竞品凭空更准约 2 倍, cross_check 假红。
    """

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", **kwargs):
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def __call__(self, pred, target, weight=None, **kwargs):
        out_dtype = pred.dtype
        p = pred.float()
        t = target.float()
        w = weight.float() if weight is not None else torch.ones_like(p)

        ce_sum = (-torch.log(p) * t * w).sum(dim=-1, keepdim=True)
        fw_sum = (self.alpha * torch.exp(self.gamma * torch.log(1.0 - p)) * t).sum(
            dim=-1, keepdim=True
        )
        y = (ce_sum * fw_sum).expand_as(p).contiguous()
        # 浮点输出必须 cast 回 NPU 输出 dtype, 否则竞品天然更准, ratio 失真
        return [y.to(out_dtype)]


class SoftmaxFocalLossKernelSpec:
    """kernel + geir 共用。golden 收 numpy.ndarray, 返 numpy.ndarray。

    参数名取自 op_host/softmax_focal_loss_def.cpp: pred / target / weight。
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


# 【不存在】aclnn 通路: canndev 老树 ops/built-in/op_api 与新树 ops/ 下均无
#   softmax_focal_loss 的 op_api 实现, 也无 docs/aclnnSoftmaxFocalLoss.md(01 §3.3)。
# 【不存在】e2e(torch) 通路: torch_npu 2.10.0 的 libtorch_npu.so 无 aclnnSoftmaxFocalLoss
#   符号 —— aclnn 本就不存在, torch 无从下发。
# 【不存在】tf / onnx / caffe 通路: canndev ops/built-in/framework/ 下无本算子 adapter。
