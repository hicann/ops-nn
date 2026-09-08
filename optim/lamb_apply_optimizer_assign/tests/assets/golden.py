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

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None


__golden__ = {
    "kernel": {"lamb_apply_optimizer_assign": "lamb_apply_optimizer_assign_golden"}
}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量
    运算会被抬到双精度，而算子在 fp32 上算（A2 的 TBE compute 里 dtype='float32'，
    arch35 DAG 的计算类型 U = float）。numpy 只用于取值与 dtype 转换。"""
    # 标量落成 **0 维 float64**: torch 类型提升里 0 维不抬 dim>0 张量的档,
    # 故数据 fp32 时结果仍 fp32、被 Promote 成 fp64 时标量自动跟到 fp64。
    return tuple(
        torch.from_numpy(np.asarray(x, "float64").reshape(-1)[:1])[0] for x in xs
    )


def _t(x):
    """**跟随 TTK 下发的 dtype 计算，不要强制降到 float32。**

    cross_check 下 TTK 走 golden_mode=Promote，按 DTYPE_PROMOTE_MAP 抬一档下发
    (fp16/bf16->float32, fp32->float64)，以满足精度标准 §4.5「更高精度的 CPU 实现为真值」。
    原先无条件 astype("float32") 把 Promote 抬上来的 fp64 又降回 fp32：golden 与三方腿
    逐位相等 -> 双标杆塌成单标杆 -> 三比值分母夹到 §4.5.1 的 err -> 有量纲的 RMSE
    比值随输出量级线性放大而假红。

    fp64 照收，其余一律 float32（与改动前一致，算子在 fp32 上算），非 cross_check 行为不变。
    """
    a = np.asarray(x)
    return torch.from_numpy(a if a.dtype == np.float64 else a.astype("float32"))


def lamb_apply_optimizer_assign_golden(
    grad,
    inputv,
    inputm,
    input3,
    mul0_x,
    mul1_x,
    mul2_x,
    mul3_x,
    add2_y,
    steps,
    do_use_weight,
    weight_decay_rate,
    **kwargs,
):
    """Golden for LambApplyOptimizerAssign. Params follow lamb_apply_optimizer_assign_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch tensor ops (torch.add/torch.addcmul/torch.sqrt) instead of a
    hand-written numpy formula: red line R3 requires the golden to be a competitor-operator
    composition, and a naive numpy expression tends to make exactly the same rounding mistakes
    as the kernel under test, which would disguise a precision shortfall as a pass.
    """
    dt = grad.dtype
    g, v, m, w = [_t(x) for x in (grad, inputv, inputm, input3)]
    b1, omb1, b2, omb2, eps, t, du, wd = _scalars(
        mul0_x, mul1_x, mul2_x, mul3_x, add2_y, steps, do_use_weight, weight_decay_rate
    )
    # 两步(先乘后加)拼接,不用 addcmul/add(alpha=) 的融合形式:后者可能走 FMA 单次舍入,
    # 与算子定义的「Muls 再 Add」两步舍入不是同一个运算序列。golden 要如实转写定义。
    next_v = v * b2 + (g * g) * omb2
    next_m = m * b1 + g * omb1
    # 偏差校正按内核的算法写：arch35 DAG 用 Log/Mul/Exp 三条指令实现幂
    #   LnB1=Log(B1); ExpArg1=Mul(LnB1,Steps); B1Steps=Exp(ExpArg1)
    #   NegB1Steps=Muls(B1Steps,-1); B1corr=Adds(NegB1Steps,1)
    # 而不是 b1 ** t。两者数学等价、浮点下不等价：Log/Exp 各是标称 1 ULP 的单指令，
    # 且 b1 接近 1 时 ln(b1) 有相消损失，直接幂运算得不到同一个中间量。
    b1_corr = 1.0 + (-1.0) * torch.exp(torch.log(b1) * t)
    b2_corr = 1.0 + (-1.0) * torch.exp(torch.log(b2) * t)
    update = (next_m / b1_corr) / (torch.sqrt(next_v / b2_corr) + eps) + w * wd * du
    return [
        update.numpy().astype(dt),
        next_v.numpy().astype(dt),
        next_m.numpy().astype(dt),
    ]


# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# golden     = CPU 真值，如实转写算子定义（两步拼接，不用融合算子）
# third_party= 三方标杆，用 torch 的自然形式（含融合算子）在设备侧跑，供 cross_check 比对
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。

    **不抬精度**: 三方标杆必须按算子自身 dtype 计算。此前统一 .to(float32) 会让三方与
    走 Promote(fp32) 的 golden 逐位相等, cross_check 的分母塌到 safe_div 的 small_value
    地板, 判据退化成"NPU 与 fp32 参照的绝对误差", 随输出量级线性放大而必红。
    仅 bf16 需还原载体(torch 不收 ml_dtypes 的 bf16 视图), 其余保持原 dtype。
    """
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return (
        t.to(torch.float32)
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        else t
    )


def _tp_s(x):
    return _tp_t(x).reshape(-1)[0]


class _LambApplyOptimizerAssignCompose:
    def __call__(
        self,
        grad,
        inputv,
        inputm,
        input3,
        mul0_x,
        mul1_x,
        mul2_x,
        mul3_x,
        add2_y,
        steps,
        do_use_weight,
        weight_decay_rate,
        **kwargs,
    ):
        g, v, m, w = (_tp_t(t) for t in (grad, inputv, inputm, input3))
        b1, omb1, b2, omb2, eps, t, du, wd = (
            _tp_s(x)
            for x in (
                mul0_x,
                mul1_x,
                mul2_x,
                mul3_x,
                add2_y,
                steps,
                do_use_weight,
                weight_decay_rate,
            )
        )
        # 与内核同运算序列(dag.h:66 `Add(NV1, NV2)`, NV1/NV2 各自独立乘出来):
        # 不能用 addcmul 的 FMA 单次舍入, 否则竞品凭空更准, ratio 误判内核。
        next_v = v * b2 + (g * g) * omb2
        next_m = m * b1 + g * omb1
        b1_corr = 1.0 - torch.pow(b1, t)
        b2_corr = 1.0 - torch.pow(b2, t)
        update = (next_m / b1_corr) / (torch.sqrt(next_v / b2_corr) + eps) + w * wd * du
        return [update, next_v, next_m]


# ---------------------------------------------------------------------------
# 三方(GPU)腿按内核算法转写: 入口复刻 Cast<U, T, 0>(窄类型加宽到内核计算类型),
# 出口复刻 Cast<T, U, 1> / NarrowStore(窄回算子输出 dtype)。两者**必须成对**:
#   * 缺入口加宽 -> 三方在窄类型上逐步截断, 与融合内核(中间量全程 fp32)不是同一算法;
#     A100 实测 fp16 下 300*300 直接得 inf, 而 (a*a)/a 真值 300 明明存得下。
#   * 缺出口窄回 -> 三方停在 fp32, 与走 TTK Promote(fp16->fp32) 的 golden 逐位相等,
#     双标杆塌成单标杆, 三比值分母夹到精度标准 §4.5.1 的 err, 有量纲的 RMSE 比值随
#     输出量级线性放大而假红(真机实测: 不窄回 rmse 比值 2348.6, 窄回后 1.0000)。
# 整型不动: 内核对 int 也是原生/int32 累加, 走一趟 fp32 会把 >2^24 抹掉低位。
# 只作用于 third_party(GPU); CPU golden 由 TTK 的 Promote 单独喂高精度输入, 不受影响。
# ---------------------------------------------------------------------------


def _kf_widen(a, seen):
    if isinstance(a, (list, tuple)):
        return type(a)(_kf_widen(x, seen) for x in a)
    if isinstance(a, torch.Tensor):
        if a.dtype in (torch.float16, torch.bfloat16):
            seen.append(a.dtype)
            return a.float()
        return a
    if isinstance(a, np.ndarray) or (_KF_BF16 is not None and hasattr(a, "dtype")):
        n = np.asarray(a)
        if n.dtype.kind == "V" and _KF_BF16 is not None:
            n = n.view(_KF_BF16)
        if _KF_BF16 is not None and n.dtype == _KF_BF16:
            seen.append(torch.bfloat16)
            return n.astype(np.float32)
        if n.dtype == np.float16:
            seen.append(torch.float16)
            return n.astype(np.float32)
        return a
    return a


def _kf_narrow(o, dt):
    if isinstance(o, (list, tuple)):
        return type(o)(_kf_narrow(x, dt) for x in o)
    if isinstance(o, torch.Tensor) and o.is_floating_point():
        return o.to(dt)
    return o


class _TpKernelFaithful:
    _INNER = _LambApplyOptimizerAssignCompose

    def __call__(self, *args, **kwargs):
        seen = []
        wa = [_kf_widen(a, seen) for a in args]
        wk = {k: _kf_widen(v, seen) for k, v in kwargs.items()}
        outs = self._INNER()(*wa, **wk)
        return outs if not seen else _kf_narrow(outs, seen[0])


# TTK 服务端按 __call__ 的**签名**做入参绑定(remote/server/executor.py::_bind ->
# _function_param_names)。包装类若只写 *args/**kwargs, 服务端取不到形参名, 会退化成
# 位置传参、同时又传同名关键字 -> "got multiple values for argument ..." 而三方腿整个不可用。
# 故把内层 compose 的签名透传出去。
try:
    _TpKernelFaithful.__call__.__signature__ = _kf_inspect.signature(
        _LambApplyOptimizerAssignCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class LambApplyOptimizerAssignKernelSpec:
    golden = lamb_apply_optimizer_assign_golden
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_apply_optimizer_assign": "LambApplyOptimizerAssignKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# TensorFlow: 有 framework 的 tf_plugin(OriginOpType "LambApplyOptimizerAssign"), 但**三方腿仍用 torch, 不补 tf**:
#   该 OriginOpType 是华为自定义 TF 类型, stock TF 里没有对应算子, tf 腿只能用 TF 张量运算
#   拼出等价语义。拼接体不是"TF 的那一个算子", 精度上相对 torch 拼接没有增量, 性能腿
#   (--xpu-perf)量的更是一串算子的总时延, 与被测单算子不可比 → 补了也用不上。
#   tf 通路的连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证(TTK 的 tf 通路是 e2e 前端,
#   NPU 侧需 Ascend TF adapter, 当前环境未装)。
# e2e / ONNX / 融合 pass: 均未交付。
