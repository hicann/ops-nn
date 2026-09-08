#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch

import inspect as _kf_inspect

try:
    from ml_dtypes import bfloat16 as _KF_BF16
except ImportError:
    _KF_BF16 = None


__golden__ = {"kernel": {"lamb_update_with_lr_v2": "lamb_update_with_lr_v2_golden"}}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量
    运算会被抬到双精度，而算子在 fp32 上算（A2 的 TBE compute 里 dtype='float32'，
    arch35 DAG 的计算类型 U = float）。numpy 只用于取值与 dtype 转换。"""
    # 标量落成 **0 维 float64** 张量: torch 类型提升里 0 维张量不会抬 dim>0 张量的档，
    # 故数据 fp32 时结果仍 fp32(与改动前一致)，数据被 Promote 成 fp64 时标量自动跟到 fp64。
    return tuple(
        torch.from_numpy(np.asarray(x, "float64").reshape(-1)[:1])[0] for x in xs
    )


def _t(x):
    """**跟随 TTK 下发的 dtype 计算，不要强制降档、也不要自行抬到 fp64。**

    cross_check 下 TTK 走 golden_mode=Promote，按 DTYPE_PROMOTE_MAP 抬一档下发
    (fp16/bf16->float32, fp32->float64)，以满足精度标准 §4.5「更高精度的 CPU 实现为真值」。
    golden 按下发的 dtype 计算并输出即可，回落到算子输出 dtype 由框架处理。

    原先无条件 astype("float32") 把 Promote 抬上来的 fp64 又降回 fp32：对 fp32 用例等于
    撤销了这一档抬升，golden 与三方腿逐位相等 -> 双标杆塌成单标杆 -> 三比值分母夹到
    §4.5.1 的 err -> 有量纲的 RMSE 比值随输出量级线性放大而假红。
    """
    a = np.asarray(x)
    return torch.from_numpy(a if a.dtype == np.float64 else a.astype("float32"))


def _fp32_div(a, b):
    """IEEE-754 fp32 scalar division through torch.div (x/0 -> +-inf, 0/0 -> nan, never raises)."""
    return torch.div(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    ).item()


def lamb_update_with_lr_v2_golden(x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
    """Golden for LambUpdateWithLrV2. Params follow lamb_update_with_lr_v2_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch ops (torch.div for the ratio, torch tensor arithmetic for the
    update) instead of a hand-written numpy formula: red line R3 requires the golden to be a
    competitor-operator composition, and a naive numpy expression tends to make exactly the
    same rounding mistakes as the kernel under test, which would disguise a precision
    shortfall as a pass.
    """
    dt = x4.dtype
    a, b, lr, gy, se = _scalars(x1, x2, x3, greater_y, select_e)
    upd, param = _t(x4), _t(x5)
    # Match kernel Vec::Div<float> (arch35 DivAlgo::INTRINSIC): IEEE-754 fp32 division.
    # b<=gy stays on select_e (kernel Select); when b>gy and b==0 the kernel outputs inf
    # (v2 has NO clip), so mirror that instead of raising ZeroDivisionError.
    inner = _fp32_div(a, b) if b > gy else se
    ratio = inner if a > gy else se
    return [(param - lr * ratio * upd).numpy().astype(dt)]


# ----------------------------------------------------------------------------
# TTK 新版 spec 注册（kernel 通路）: 在保留原 golden 的基础上补三方标杆能力。
# golden     = CPU 真值，如实转写算子定义（两步拼接，不用融合算子）
# third_party= 三方标杆，用 torch 的自然形式（含融合算子）在设备侧跑，供 cross_check 比对
# ----------------------------------------------------------------------------
_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


try:
    from ml_dtypes import bfloat16 as _TP_BF16
except ImportError:
    _TP_BF16 = None


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备。

    **不替 torch 决定精度**: 原样把张量交给 torch, 是否在内部抬到 fp32 由 torch 的算子
    实现自行决定(CUDA 上不少算子对 fp16 输入会用 fp32 累加)。三方标杆的价值在于
    CPU/NPU/GPU 三套**独立实现**互相制约——一旦按内核的算法(fp32 中间量)去复刻, 它就不再
    独立、再也发现不了算法层面的缺陷; 精度标准 §5.3.1 第 1 类要的就是「竞品的自然实现」。
    仅 bf16 需还原载体(torch 不收 ml_dtypes 的 bf16 视图), 其余(含整型)保持原 dtype
    —— 整型必须整型算, 走 fp32 会丢低位。
    """
    if isinstance(x, torch.Tensor):
        return x
    a = np.asarray(x)
    # bf16 用 ml_dtypes 承载, torch 不认其 void 载体, 按位 view 回 bf16(同宽无损)。
    # 这是**载体还原**, 不是精度干预; 其余(含整型)一律原样, 整型必须整型算 ——
    # 走一趟 fp32 会把 >2^24 的 int32 抹掉低位(见 #85 的成因)。
    if a.dtype.kind == "V" and _TP_BF16 is not None:
        a = a.view(_TP_BF16)
    if _TP_BF16 is not None and a.dtype == _TP_BF16:
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(a)


def _tp_s(x):
    return _tp_t(x).reshape(-1)[0]


class _LambUpdateWithLrV2Compose:
    def __call__(self, x1, x2, x3, x4, x5, greater_y, select_e, **kwargs):
        a, b, lr, gy, se = (_tp_s(v_) for v_ in (x1, x2, x3, greater_y, select_e))
        upd, param = _tp_t(x4), _tp_t(x5)
        inner = torch.div(a, b) if bool(b > gy) else se
        ratio = inner if bool(a > gy) else se
        return [param - lr * ratio * upd]


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
    _INNER = _LambUpdateWithLrV2Compose

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
        _LambUpdateWithLrV2Compose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class LambUpdateWithLrV2KernelSpec:
    golden = lamb_update_with_lr_v2_golden
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_update_with_lr_v2": "LambUpdateWithLrV2KernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# e2e / ONNX / 融合 pass: 均未交付。
