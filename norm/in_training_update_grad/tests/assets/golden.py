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
TTK TestSpec for in_training_update_grad (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（rsqrt/广播乘/sum 规约），fp64 计算（理由见下）
    third_party  —— 三方标杆，torch 拼接在远端 GPU 上执行（fp32，竞品自然精度）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

为什么 golden 必须 fp64（与"golden 别抬 fp64"的一般规则不同，特事特办）：
本算子是大规约（D*H*W 可至数万），内核用 fp32+Kahan 补偿、精度已达 fp32 地板；
fp32 朴素累加的 golden 在大/对消规约（如 D=80000）上自带 ~eps*kappa 的误差，
比内核还差（实测差 4~5 倍），拿它当参照会把更准的内核误判成不达标——
故 golden 的精度档必须严格高于内核。档位不由 golden 自己硬抬,而是照
ttk_golden_logic.md 三/四: 三方档由 TTK Promote 抬(golden 零 cast), 泛化档跟内核的
加宽行为(f16/bf16 抬 f32, f32 不再加宽)——见 _work_dtype。

Canonical IO order (in_training_update_grad_def.cpp):
    inputs : dy, x, variance, mean   (NDC1HWC0 6D: (N, D, C1, H, W, C0))
    outputs: res_gamma, res_beta     (fp32)
    attrs  : 无（eps=1e-6 为算子定义常量）

    x_norm    = (x - mean) * rsqrt(variance + 1e-6)   # mean/variance 按 D,H,W 广播
    res_gamma = sum_{D,H,W} dy * x_norm               # keepdims，空间维 -> 1
    res_beta  = sum_{D,H,W} dy                        # keepdims
空规约（D/H/W == 0）两个输出都为 0.0（空集求和）。
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

_EPS = 1e-6
_REDUCE_AXES = (1, 3, 4)  # D, H, W


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


def _compute(dy, x, variance, mean, **_):
    """torch.Tensor 进 / 出（档位由 _work_dtype 定），返回 [res_gamma, res_beta]，顺序照 def.cpp。"""
    work = _work_dtype(dy, x, variance, mean)
    dy_t = dy.to(work)
    x_t = x.to(work)
    var_t = variance.to(work)
    mean_t = mean.to(work)

    rstd = torch.rsqrt(var_t + _EPS)
    x_norm = (x_t - mean_t) * rstd
    res_gamma = (dy_t * x_norm).sum(dim=_REDUCE_AXES, keepdim=True)
    res_beta = dy_t.sum(dim=_REDUCE_AXES, keepdim=True)
    return [res_gamma, res_beta]


def _tp_widen(t):
    """三方腿的计算档: 跟 NPU 的加宽行为(ttk_golden_logic.md 三/四 的 GPU 列)。

    内核把 f16/bf16 抬到 f32 做中间量, f32 不再加宽。本算子两个输出恒为 fp32,
    内核不窄回, 故三方也不窄回。此前 res_beta = dy.sum() 直接留在 fp16,
    dtype 与 NPU 输出(fp32)都对不上。
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


class _IntugCompose:
    """三方标杆：torch 拼接在远端 GPU 执行，fp32（竞品自然精度，不抬 fp64）。

    本算子内核是 fp32+Kahan，比 fp32 朴素竞品更准——cross_check 比值 <1 属预期
    （比值小表示 NPU 误差更小，PASS）；若竞品抬 fp64 则分母趋零、比值爆表，
    会把更准的内核误判成缺陷（三方须同精度对等）。
    计算档由 `_tp_widen` 跟随内核(f16/bf16 抬 f32)；两个输出恒 fp32、内核不窄回，故三方也不窄回。
    """

    def __init__(self, **_):
        pass

    def __call__(self, dy, x, variance, mean, **_):
        dyw, xw = _tp_widen(dy), _tp_widen(x)
        rstd = torch.rsqrt(_tp_widen(variance) + _EPS)
        x_norm = (xw - _tp_widen(mean)) * rstd
        res_gamma = (dyw * x_norm).sum(dim=_REDUCE_AXES, keepdim=True)
        res_beta = dyw.sum(dim=_REDUCE_AXES, keepdim=True)
        return [res_gamma, res_beta]


def _inject_nonfinite(dy, x, variance, mean, testcase_name=""):
    """DFX-nonfinite 档的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(ttk/utilities/data.py `_digitize_inf_nan`), 写了也造不出
    非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入**数据面**输入(dy, x), 不注入权重/统计量(variance, mean):
    后者是逐通道广播量, 注入会让整通道输出非有限, 掩盖"非有限值沿计算链如何传播"这一档
    真正要看的东西。位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan。元素数 < 3 不注入。

    预期行为: 按 IEEE 语义自然传播, 仍走正常精度判据, 不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return dy, x, variance, mean
    _t0 = np.ascontiguousarray(dy).copy() if dy is not None else None
    _t1 = np.ascontiguousarray(x).copy() if x is not None else None
    _t2 = np.ascontiguousarray(variance).copy() if variance is not None else None
    _t3 = np.ascontiguousarray(mean).copy() if mean is not None else None
    if _t0 is not None and _t0.size >= 3:
        _f = _t0.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    if _t1 is not None and _t1.size >= 3:
        _f = _t1.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    return _t0, _t1, _t2, _t3


class InTrainingUpdateGradSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray](档位见 _work_dtype)。"""

    def golden(dy, x, variance, mean, **kwargs):
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(dy)),
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(variance)),
            torch.from_numpy(np.ascontiguousarray(mean)),
            **kwargs,
        )
        # 出口跟随内核的窄回行为: 本算子两个输出恒为 fp32, 内核不窄回,
        # 故直接给计算 dtype(Promote 档即 fp64)。
        return [o.numpy() for o in outs]

    def customize_inputs(dy, x, variance, mean, **kwargs):
        return _inject_nonfinite(dy, x, variance, mean, kwargs.get("testcase_name", ""))

    third_party = {"torch": _IntugCompose}
    tolerance = _TOL


def in_training_update_grad_golden(dy, x, variance, mean, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(InTrainingUpdateGradSpec.golden(dy, x, variance, mean, **kwargs))


__spec__ = {"in_training_update_grad": "InTrainingUpdateGradSpec"}
__golden__ = {"kernel": {"in_training_update_grad": "in_training_update_grad_golden"}}

# 【不存在】aclnn 通路：canndev 无 op_api/aclnnINTrainingUpdateGrad（01 §3.3，纯 GE 图算子）。
# 【不存在】e2e 通路：torch_npu 二进制 0 引用（strings libtorch_npu.so 无 INTraining 子串）；
#   该算子仅由 GE 图在做 InstanceNorm 训练反向时内部构造，torch 侧无直达通路。
# 【不存在】tf / onnx / caffe 通路：canndev framework 插件全树 grep 0 命中（01 §3.3）。
