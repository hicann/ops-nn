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
    golden       —— 真值，torch 算子拼接（sum/sqrt/clamp/逐点乘除），档位见下
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

精度档位（ttk_golden_logic.md 三/四）：golden **自己不抬精度**。
  - 三方档(cross_check)由 TTK 的 Promote 把入口 dtype 整体抬一档,golden 零 cast;
    自行 `.to(float64)` 不但是重复动作,还会掩盖 Promote 空转(TTK 为此留了 warning)。
  - 两方泛化档 TTK 不提升,规范要求跟 NPU 的加宽行为——内核把 f16/bf16 抬到 f32 做
    中间量、f32 不再加宽,golden 照此由 `_work_dtype` 决定,出口按下发 dtype 窄回。
  dx = dy - y*s 是对消差,对 s 的归约误差敏感;真值的精度由上面的档位规则保证,
  不靠 golden 自己硬抬。

Canonical IO order (l2_normalize_grad_def.cpp):
    inputs : x, y, dy（同 dtype）
    outputs: dx（同 x dtype/shape）
    attrs  : dim(OPTIONAL ListInt={}), eps(OPTIONAL float=1e-4)
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
    """折负 + 去重 + 排序，返回轴元组；空元组表示不归约（对齐 ascend910b 的 GE 通路语义）。

    重复/乱序/正负混写必须在这里收敛：numpy 与 torch 对重复轴是直接报错的
    （`duplicate value in 'axis'` / `dim N appears multiple times`），不规范化会变成假红。
    """
    if dim is None:
        return ()
    vals = list(dim) if isinstance(dim, (list, tuple, np.ndarray)) else [dim]
    axes = set()
    for v in vals:
        a = int(v)
        if a < 0:
            a += rank
        axes.add(a)
    return tuple(sorted(axes))


def _reduce_pair(a, b, axes):
    """按 axes 求和（keepdim）；axes 为空表示不归约，逐元素返回。

    ⚠️ 不能把空元组交给 torch.sum(dim=())——它会被当成“对所有维求和”，真值全错。
    """
    if axes:
        return a.sum(dim=axes, keepdim=True), b.sum(dim=axes, keepdim=True)
    return a, b


def _attr(kwargs, name, default):
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    return default if v is None else v


def _work_dtype(*tensors):
    """计算档位:只向上兜底 —— f16/bf16 抬 f32;任一输入是 f64(Promote 抬档)则全 f64。

    golden 自己不做 Promote:三方档(cross_check)由 TTK 抬(入口 dtype 已整体高一档),
    这里再 `.to(float64)` 是重复动作、还会掩盖 Promote 空转;两方泛化档 TTK 不提升,
    规范(ttk_golden_logic.md 三/四)要求跟 NPU 的加宽行为——内核把 f16/bf16 抬到 f32
    做中间量,f32 不再加宽。
    """
    if any(t.dtype == torch.float64 for t in tensors):
        return torch.float64
    return torch.float32


def _compute(x, y, dy, **kwargs):
    """torch.Tensor 进 / 出（档位由 _work_dtype 定），返回 [dx]，顺序照 def.cpp。"""
    axes = _resolve_axis(_attr(kwargs, "dim", ()), x.dim())
    eps = float(_attr(kwargs, "eps", 1e-4))

    work = _work_dtype(x, y, dy)
    xf = x.to(work)
    yf = y.to(work)
    dyf = dy.to(work)

    # ── 以下全部为 torch 库算子拼接，不手写 numpy 数值公式（红线 R3）──
    sq, s = _reduce_pair(xf * xf, yf * dyf, axes)
    n = torch.clamp(torch.sqrt(sq), min=eps)
    dx = (dyf - yf * s) / n
    return [dx]


def _tp_widen(t):
    """三方腿的计算档: **跟 NPU 的加宽行为**(ttk_golden_logic.md 三/四 的 GPU 列)。

    内核把 f16/bf16 抬到 f32 做中间量, f32 不再加宽; 三方腿必须同步, 出口再窄回。
    不同步的后果是三方被人为劣化 —— 它把整条归约留在 f16 算, 误差远大于内核,
    cross_check 的分母虚高、比值系统性偏小, 内核有缺陷也照样 PASS。
    (L2 实测: fp16 档 mare 比值中位 0.0135, fp32 档 0.4697, 差 35 倍。)
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


class _L2NormalizeGradCompose:
    """三方标杆：torch 拼接在远端 GPU 执行。不抬 fp64——否则分母趋零、cross_check
    比值爆表，会把内核误判成缺陷；三方须与 NPU 同精度对等。
    参数名与 def.cpp 逐字一致（x/y/dy/dim/eps）。
    计算档由 `_tp_widen` 跟随内核(f16/bf16 抬 f32)，出口按 dy.dtype 窄回。
    """

    def __init__(self, *, dim=(), eps=1e-4, **_):
        self.dim = dim
        self.eps = float(eps)

    def __call__(self, x, y, dy, **_):
        axes = _resolve_axis(self.dim, x.dim())
        xw, yw, dyw = _tp_widen(x), _tp_widen(y), _tp_widen(dy)
        sq, s = _reduce_pair(xw * xw, yw * dyw, axes)
        n = torch.clamp(torch.sqrt(sq), min=self.eps)
        return [((dyw - yw * s) / n).to(dy.dtype)]


def _inject_nonfinite(x, y, dy, testcase_name=""):
    """DFX-nonfinite 档的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(ttk/utilities/data.py `_digitize_inf_nan`), 写了也造不出
    非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入**数据面**输入(x, dy), 不注入权重/统计量(y):
    后者是逐通道广播量, 注入会让整通道输出非有限, 掩盖"非有限值沿计算链如何传播"这一档
    真正要看的东西。位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan。元素数 < 3 不注入。

    预期行为: 按 IEEE 语义自然传播, 仍走正常精度判据, 不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return x, y, dy
    _t0 = np.ascontiguousarray(x).copy() if x is not None else None
    _t1 = np.ascontiguousarray(y).copy() if y is not None else None
    _t2 = np.ascontiguousarray(dy).copy() if dy is not None else None
    if _t0 is not None and _t0.size >= 3:
        _f = _t0.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    if _t2 is not None and _t2.size >= 3:
        _f = _t2.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    return _t0, _t1, _t2


class L2NormalizeGradSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray](档位见 _work_dtype)。"""

    def golden(x, y, dy, **kwargs):
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(dy)),
            **kwargs,
        )
        # 出口跟随下发 dtype 窄回(Promote 档下发即 fp64, 此处自然是 no-op)。
        return [o.numpy().astype(x.dtype) for o in outs]

    def customize_inputs(x, y, dy, **kwargs):
        return _inject_nonfinite(x, y, dy, kwargs.get("testcase_name", ""))

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
