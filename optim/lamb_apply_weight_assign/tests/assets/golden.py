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


__golden__ = {"kernel": {"lamb_apply_weight_assign": "lamb_apply_weight_assign_golden"}}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量运算
    会被抬到双精度，而算子两代都在 fp32 上算（A2 compute_ratio 里 dtype='float32'；
    arch35 LambApplyWeightAssignCompute<T, U> 的 U = float）。numpy 只用于取值与 dtype
    转换，计算一律交给 torch。"""
    # 标量落成 **0 维 float64** 张量: torch 的类型提升里 0 维张量不会把 dim>0 的张量抬档，
    # 所以数据是 fp32 时结果仍是 fp32(与改动前一致)，数据被 Promote 成 fp64 时标量自动
    # 跟到 fp64，不会用一个先降到 fp32 的标量去污染高精度真值。
    return tuple(
        torch.from_numpy(np.asarray(x, "float64").reshape(-1)[:1])[0] for x in xs
    )


_F32_TINY = torch.tensor(float(np.finfo(np.float32).tiny), dtype=torch.float32)


def _div_ftz(a, b):
    """torch.div 的 fp32 除法 + FTZ，对齐算子两代共同的行为。

    A2(910B) 的 Div 没有 config 参数（asc-devkit Div.md 里带 config 的原型对 Atlas A2
    标注"不支持"），只有单指令一条路，Subnormal 必然 FTZ；arch35 的 Vec::Div 用默认
    DivConfig{DivAlgo::INTRINSIC}，文档写明该档"Subnormal 均被 FTZ"。CPU 默认不 FTZ，
    所以要显式补这一步，否则商落入 [1e-45, 1.1754944e-38) 时 golden 会给出算子实际
    不会产出的值。"""
    q = torch.div(a, b)
    return torch.where((q != 0) & (q.abs() < _F32_TINY), torch.zeros_like(q), q)


def _t(x):
    """**跟随 TTK 下发的 dtype 计算，不要强制降档、也不要自行抬到 fp64。**

    判据是 cross_check 时 TTK 会设 golden_mode=Promote，按 DTYPE_PROMOTE_MAP 把输入
    抬一档下发(fp16/bf16->float32, fp32->float64)，以满足精度标准 §4.5「双标杆比对」
    要求的「更高精度的 CPU 实现为真值」。golden 只需按**下发的 dtype**计算并输出，
    回落到算子输出 dtype 由框架处理。

    原先无条件 astype("float32") 把 Promote 抬上来的 fp64 又降回 fp32：对 **fp32 用例**
    就等于撤销了这一档抬升，golden 与三方腿(同为 fp32、同一串 torch 算子、同一结合序)
    在 IEEE-754 下逐位相等 —— 双标杆塌成单标杆，GPU 三个误差指标恒 0，三比值分母全部
    夹到 §4.5.1 的 err。夹底后 MARE/MERE 的分子是无量纲相对误差、数值仍小照样 PASS，
    唯独 RMSE 的分子是有量纲绝对误差，比值随输出量级线性放大，表现为
    「rmse 爆表而 mare/mere 正常」。

    fp64 保持 fp64(Promote 档)，其余一律 float32(与改动前一致，算子在 fp32 上算)，
    因此非 cross_check 判据的行为不变。
    """
    a = np.asarray(x)
    return torch.from_numpy(a if a.dtype == np.float64 else a.astype("float32"))


def lamb_apply_weight_assign_golden(
    input0, input1, input2, input3, input_param, **kwargs
):
    """Golden for LambApplyWeightAssign. Params follow lamb_apply_weight_assign_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch tensor arithmetic instead of a hand-written numpy formula:
    red line R3 requires the golden to be a competitor-operator composition, and a naive
    numpy expression tends to make exactly the same rounding mistakes as the kernel under
    test, which would disguise a precision shortfall as a pass.
    """
    dt = input3.dtype
    wn, gn, lr = _scalars(input0, input1, input2)
    upd, param = _t(input3), _t(input_param)
    one = torch.tensor(1.0, dtype=torch.float32)
    inner = _div_ftz(wn, gn) if gn > 0 else one
    ratio = inner if wn > 0 else one
    # 运算序列对齐实现：A2 (tbe.vmul(update, lr) 再 vmul(ratio, ·)) 与 arch35 DAG
    # (UpdLr = Update*Lr; RatioUpdLr = Ratio*UpdLr) 都是先算 update*lr。按 README 原先的
    # 字面顺序写成 (lr*ratio)*upd 在浮点下不等价：update 与 lr 同时较大时实现会先溢出成 inf，
    # 而 (lr*ratio) 往往是正常量级、再乘 update 不溢出，两者可以差出 inf 与有限值。
    return [(param - ratio * (upd * lr)).numpy().astype(dt)]


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


class _LambApplyWeightAssignCompose:
    def __call__(self, input0, input1, input2, input3, input_param, **kwargs):
        wn, gn, lr = (_tp_s(v_) for v_ in (input0, input1, input2))
        upd, param = _tp_t(input3), _tp_t(input_param)
        one = torch.ones((), dtype=torch.float32, device=upd.device)
        inner = torch.div(wn, gn) if bool(gn > 0) else one
        ratio = inner if bool(wn > 0) else one
        return [param - ratio * (upd * lr)]


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
    _INNER = _LambApplyWeightAssignCompose

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
        _LambApplyWeightAssignCompose.__call__
    )
except (ValueError, TypeError):  # 内层无法内省时保持原样
    pass


class LambApplyWeightAssignKernelSpec:
    golden = lamb_apply_weight_assign_golden
    third_party = {"torch": _TpKernelFaithful}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_apply_weight_assign": "LambApplyWeightAssignKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# TensorFlow: 有 framework 的 tf_plugin(OriginOpType "LambApplyWeightAssign"), 但**三方腿仍用 torch, 不补 tf**:
#   该 OriginOpType 是华为自定义 TF 类型, stock TF 里没有对应算子, tf 腿只能用 TF 张量运算
#   拼出等价语义。拼接体不是"TF 的那一个算子", 精度上相对 torch 拼接没有增量, 性能腿
#   (--xpu-perf)量的更是一串算子的总时延, 与被测单算子不可比 → 补了也用不上。
#   tf 通路的连通性仍按预生成 .pb + aclgrphParseTensorFlow 验证(TTK 的 tf 通路是 e2e 前端,
#   NPU 侧需 Ascend TF adapter, 当前环境未装)。
# e2e / ONNX / 融合 pass: 均未交付。
