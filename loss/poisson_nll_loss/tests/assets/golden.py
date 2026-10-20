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
        # 按 NPU 的加宽行为同步 cast, 出口对应窄回 ——  四格表「浮点+三方」那一行、V2 第三行。
        # 本算子内核 U = fp32(op_kernel/arch35 通篇 RegTensor<float>/LocalTensor<float>),
        # 而 torch 的 poisson_nll_loss 是 exp/log/mul/sub + 归约的组合, 算子之间每步落回 T
        # ⇒ 三方腿若停在 fp16/bf16 就是「逐步截断的实现」当竞品, 误差远大于内核
        #   cross_check 分母虚高、比值系统性偏小 ⇒ **内核有缺陷也照样 PASS**(判据被放水)。
        # fp32/fp64 不动(U = T, torch 原生即是), 否则竞品凭空变准 ⇒ 假红。
        # 同批的 MSELossV2 已按此修, 本算子当时漏了。
        kw = dict(
            log_input=self.log_input,
            full=self.full,
            eps=self.eps,
            reduction=self.reduction,
        )
        if input_x.dtype not in (torch.float16, torch.bfloat16):
            # U == T, torch 原生即是 ⇒ **不 cast**(加了就是多余 cast, 还会让竞品凭空变准 → 假红)
            return [F.poisson_nll_loss(input_x, target, **kw)]
        # 仅窄类型需要补齐「跨算子不落回」, 出口窄回 T
        y = F.poisson_nll_loss(input_x.float(), target.float(), **kw)
        return [y.to(input_x.dtype)]


def _inject_nonfinite(input_x, target, testcase_name=""):
    """DFX-nonfinite 档的定点注入：只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan：TTK 的 RandomData 会把值域里的
    inf/nan 钳到 dtype 极值(ttk/utilities/data.py `_digitize_inf_nan`), 写了也造不出
    非有限输入 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    注入位置固定(input_x 首元素 +inf、第三个 nan; target 次元素 -inf), 不随机:
    随机会让复现依赖 seed, 且小 shape 时可能一个都注不进去。元素数 < 4 不注入。

    预期行为: 内核无特判, 按 IEEE 自然传播(与 torch 一致), 仍走正常精度判据, 不是拒收档。
    注: 对齐同批 mse_loss_v2/multilabel_margin_loss 的同名钩子写法。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return input_x, target
    a, b = np.asarray(input_x).copy(), np.asarray(target).copy()
    if a.size < 4:
        return input_x, target
    fa, fb = a.reshape(-1), b.reshape(-1)
    fa[0], fa[2] = np.inf, np.nan
    fb[1] = -np.inf
    return a, b


class PoissonNllLossSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray]。"""

    def golden(input_x, target, **kwargs):
        # 用 np.asarray 而非 np.ascontiguousarray: 后者会把 0 维数组提升成 (1,),
        # 而本算子接受 rank-0 输入(def.cpp 的 shape 面与 infershape 的 SetDimNum(0) 分支),
        # 一旦泛化集补上 rank-0 用例, ascontiguousarray 会让三方腿与内核的形状语义不一致。
        # ascontiguousarray 会把 0 维数组提升成 (1,), 而本算子接受 rank-0 输入
        # (def.cpp 的 shape 面与 infershape 的 SetDimNum(0) 分支) ⇒ 保连续后按原 shape 还原,
        # 既不丢"连续"这个 from_numpy 需要的性质, 也不改变 rank 语义。
        inp = torch.from_numpy(np.ascontiguousarray(input_x).reshape(np.shape(input_x)))
        tgt = torch.from_numpy(np.ascontiguousarray(target).reshape(np.shape(target)))
        outs = _compute(inp, tgt, **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i]) if i < len(od) else o.numpy().astype(input_x.dtype)
            for i, o in enumerate(outs)
        ]

    def customize_inputs(input_x, target, **kwargs):
        # 非有限输入只能由钩子定点注入(CSV 值域写 inf/nan 会被 TTK 钳掉), 见 _inject_nonfinite。
        return _inject_nonfinite(input_x, target, kwargs.get("testcase_name", ""))

    third_party = {"torch": _PoissonNllLossCompose}
    tolerance = _TOL


def poisson_nll_loss_golden(input_x, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return PoissonNllLossSpec.golden(input_x, target, **kwargs)[0]


__spec__ = {"poisson_nll_loss": "PoissonNllLossSpec"}
__golden__ = {"kernel": {"poisson_nll_loss": "poisson_nll_loss_golden"}}

# 【不存在】aclnn 通路：A2 侧无 aclnn 接口（def ACLNNTYPE aclnn_exclude，对齐 A2 不新增）
#   见 01_requirement.md §3.2。
# 【不存在】e2e 通路：torch poisson_nll_loss 是 composite 分解，不走本融合算子；
#   torch_npu 二进制无 aclnnPoissonNllLoss 引用（dispatcher 权威判定，01 §3.2）。
# 【不存在】tf 通路：A2 canndev 无 tf_plugin 本体。
