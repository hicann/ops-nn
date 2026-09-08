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


__golden__ = {"kernel": {"lamb_next_mv": "lamb_next_mv_golden"}}


def _scalars(*xs):
    """取标量并落到 float32 torch 标量张量上。不能返回 Python float——那是 fp64，标量
    运算会被抬到双精度，而算子在 fp32 上算（A2 的 TBE compute 里 dtype='float32'，
    arch35 DAG 的计算类型 U = float）。numpy 只用于取值与 dtype 转换。"""
    # 标量落成 **0 维 float64** 张量: torch 的类型提升里 0 维张量不会把 dim>0 的张量抬档，
    # 所以数据是 fp32 时结果仍是 fp32(与改动前一致)，数据被 Promote 成 fp64 时标量自动
    # 跟到 fp64，不会用一个先降到 fp32 的标量去污染高精度真值。
    return tuple(
        torch.from_numpy(np.asarray(x, "float64").reshape(-1)[:1])[0] for x in xs
    )


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


def lamb_next_mv_golden(
    input_mul3,
    input_mul2,
    input_realdiv1,
    input_mul1,
    input_mul0,
    input_realdiv0,
    input_mul4,
    mul0_x,
    mul1_sub,
    mul2_x,
    mul3_sub1,
    mul4_x,
    add2_y,
    **kwargs,
):
    """Golden for LambNextMV. Params follow lamb_next_m_v_def.cpp (without outputs). All inputs are numpy.ndarray.

    Computed by composing torch tensor ops (torch.add/torch.sqrt) instead of a hand-written
    numpy formula: red line R3 requires the golden to be a competitor-operator composition,
    and a naive numpy expression tends to make exactly the same rounding mistakes as the
    kernel under test, which would disguise a precision shortfall as a pass.
    """
    dt = input_mul3.dtype
    g2, v, g, m, param = [
        _t(x) for x in (input_mul3, input_mul2, input_mul1, input_mul0, input_mul4)
    ]
    rd1, rd0, b1, omb1, b2, omb2, wd, eps = _scalars(
        input_realdiv1,
        input_realdiv0,
        mul0_x,
        mul1_sub,
        mul2_x,
        mul3_sub1,
        mul4_x,
        add2_y,
    )
    # 两步(先乘后加)拼接,不用 torch.add(alpha=)/addcmul 的融合形式:后者可能走 FMA 单次舍入,
    # 与算子定义的「Muls 再 Add」两步舍入不是同一个运算序列。golden 要如实转写定义。
    next_v = v * b2 + g2 * omb2
    next_m = m * b1 + g * omb1
    v_unb, m_unb = next_v / rd1, next_m / rd0
    y1 = param * wd + m_unb / torch.sqrt(v_unb + eps)
    y4 = m_unb / (torch.sqrt(v_unb) + eps)
    return [
        y1.numpy().astype(dt),
        next_m.numpy().astype(dt),
        next_v.numpy().astype(dt),
        y4.numpy().astype(dt),
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


try:
    from ml_dtypes import bfloat16 as _TP_BF16
except ImportError:
    _TP_BF16 = None


def _tp_t(x):
    """third_party 入参: kernel 通路由框架把 numpy 转成 torch 并置于目标设备(GPU)。

    仅做**载体还原**(bf16 用 ml_dtypes 承载, torch 不认其 void 视图, 按位 view 回来,
    同宽无损), 不在这里干预精度。计算精度由下面的 _tp_widen 按内核算法统一处理。
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


def _tp_widen(t):
    """把三方入参加宽到**内核的计算类型**, 复刻 op_kernel/arch35/lamb_next_m_v_dag.h 的
    `Cast<U, T, 0>`(见该文件注释: "Compute in U (=float): half casts to float")。

    这不是"给竞品放水抬精度", 而是**同算法转写**——本算子是融合 DAG, 十几步中间量全程留在
    fp32、一次都不落回 T; 而 torch 只在**单个算子内部**用 opmath=float, 算子之间每一步都把
    结果落回 fp16(A100 实测: fp16 的 300*300 直接得 inf, 而 (a*a)/a 真值 300 明明存得下)。
    不加宽就等于拿"逐步截断的实现"当竞品, 与被测内核不是同一个算法。

    整型不动: 内核对 int 也是原生/int32 累加, 走 fp32 会把 >2^24 抹掉低位。
    """
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


def _tp_narrow(outs, dt):
    """出口复刻内核的 `Cast<T, U, 1>`: 算完窄回算子输出 dtype。

    这个 cast **必须与 _tp_widen 成对出现**: 少了它, 三方停在 fp32, 会与走 TTK Promote
    (fp16->fp32) 的 golden 逐位相等 —— 双标杆塌成单标杆, 三比值分母被夹到 §4.5.1 的 err,
    有量纲的 RMSE 比值随输出量级线性放大而假红。
    """
    return [o.to(dt) if o.is_floating_point() else o for o in outs]


def _tp_s(x):
    return _tp_t(x).reshape(-1)[0]


class _LambNextMVCompose:
    def __call__(
        self,
        input_mul3,
        input_mul2,
        input_realdiv1,
        input_mul1,
        input_mul0,
        input_realdiv0,
        input_mul4,
        mul0_x,
        mul1_sub,
        mul2_x,
        mul3_sub1,
        mul4_x,
        add2_y,
        **kwargs,
    ):
        _dt = _tp_t(input_mul3).dtype  # 算子输出 dtype
        g2, v, g, m, param = (
            _tp_widen(_tp_t(t))
            for t in (input_mul3, input_mul2, input_mul1, input_mul0, input_mul4)
        )
        rd1, rd0, b1, omb1, b2, omb2, wd, eps = (
            _tp_widen(_tp_s(t))
            for t in (
                input_realdiv1,
                input_realdiv0,
                mul0_x,
                mul1_sub,
                mul2_x,
                mul3_sub1,
                mul4_x,
                add2_y,
            )
        )
        # 与内核同运算序列: dag.h:78 `Add(Mul(V,B2), Mul(G2,OmB2))` —— 两次独立乘法
        # 再一次加法, 三次舍入。**不能用 addcmul**: 它走 FMA 单次舍入, 竞品凭空更准,
        # ratio 会把内核判成缺陷(与 InstanceNormGrad 的 pow(v,-1.5) vs 连乘同一类错)。
        next_v = v * b2 + g2 * omb2
        next_m = m * b1 + g * omb1
        v_unb, m_unb = next_v / rd1, next_m / rd0
        y1 = param * wd + m_unb / torch.sqrt(v_unb + eps)
        y4 = m_unb / (torch.sqrt(v_unb) + eps)
        return _tp_narrow([y1, next_m, next_v, y4], _dt)


class LambNextMVKernelSpec:
    golden = lamb_next_mv_golden
    third_party = {"torch": _LambNextMVCompose}
    tolerance = _TOL_KERNEL


__spec__ = {"lamb_next_mv": "LambNextMVKernelSpec"}


# 通路交付情况
# 已注册: kernel + GEIR(复用 kernel spec)
# 未在 __spec__ 中注册:
# aclnn: 未交付——算子目录下无 docs/aclnn*.md。
# e2e / ONNX / 融合 pass: 均未交付。
