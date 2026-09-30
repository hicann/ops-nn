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
TTK TestSpec for instance_norm_grad (kernel / GEIR 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，torch 算子拼接（pow/sum 规约/广播逐点），fp64 计算
    third_party  —— 三方标杆，torch 拼接在远端 GPU 上执行（fp32，竞品自然精度）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

布局 NDHWC：空间维 (D,H,W) 按 (N,C) 实例规约；gamma/beta 梯度再对 N 规约（仅保留 C）。
variance 是 RAW 方差，rstd 用固定 eps=1e-6 计算；不从新鲜前向重推方差。
全部 torch 算子拼接（非 numpy 纯公式，红线 R3）。精度档不由 golden 自己硬抬:
照 ttk_golden_logic.md 三/四,三方档由 TTK Promote 抬(零 cast)、泛化档跟内核的加宽
行为——见 _work_dtype。大规约对消敏感,内核为 fp32+Kahan,已达 fp32 地板。

Canonical IO order (instance_norm_grad_def.cpp):
    inputs : dy, x, variance, mean, gamma
    outputs: pd_x, pd_gamma, pd_beta
    attrs  : 无（eps=1e-6 为算子定义常量）
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

INSTANCE_NORM_GRAD_EPS = 1e-6


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


def _sum_axes(t, axes, keepdim=True):
    """对 axes 求和；axes 为空则恒等返回（torch 会把空 dim 元组当成对所有维求和）。"""
    if not axes:
        return t
    return t.sum(dim=axes, keepdim=keepdim)


def _compute(dy, x, variance, mean, gamma, **_):
    """torch.Tensor 进 / 出（档位由 _work_dtype 定），返回 [pd_x, pd_gamma, pd_beta]，顺序照 def.cpp。"""
    nd = x.dim()
    C = x.shape[-1]
    reduce_axes = tuple(range(1, nd - 1))  # 空间轴 (D,H,W)
    m = 1
    for ax in reduce_axes:
        m *= x.shape[ax]

    work = _work_dtype(dy, x, variance, mean, gamma)
    dyf = dy.to(work)
    xf = x.to(work)

    pshape = [x.shape[0]] + [1] * (nd - 2) + [C]  # [N,1,...,1,C]
    varb = variance.to(work).reshape(pshape)
    meanb = mean.to(work).reshape(pshape)
    gshape = [1] * nd
    gshape[-1] = C
    gammab = gamma.to(work).reshape(gshape)

    rstd = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -0.5)
    rstd3 = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -1.5)

    xc = xf - meanb
    pd_xl = dyf * gammab
    # 必须用 torch 自身的规约:np.sum 会把 torch 张量转成 numpy f64 数组,后续 pd_x 也退化成
    # numpy f64,TTK 据此把图里的 dy 建成 DT_DOUBLE -> EZ3002 算子不支持。
    # rank == 2 时没有空间轴,reduce_axes 为空元组;torch 把空 dim 元组当成「对所有维求和」
    # (实测 2.10:[[0,1,2],[3,4,5]].sum(dim=())==[[15]]),而数学上对空轴集求和应为恒等。
    # 内核侧 M = 中间维乘积 = 1(空积),与恒等一致,故此处显式特判。
    pd_var = _sum_axes(-0.5 * pd_xl * xc * rstd3, reduce_axes)
    pd_mean = _sum_axes(-1.0 * pd_xl * rstd, reduce_axes)
    # m == 0 means a *spatial* axis (D/H/W) is empty: there is nothing to average over, so the
    # 1/m correction terms do not exist. The kernel's empty branch (tilingKey 500) produces an
    # empty pd_x and zeroed pd_gamma/pd_beta; matching that here keeps spatial-zero cases
    # verifiable instead of crashing the golden with a division by zero.
    inv_m = 0.0 if m == 0 else 1.0 / m
    pd_x = pd_xl * rstd + pd_var * (2.0 * inv_m) * xc + pd_mean * inv_m

    x_hat = xc * rstd
    pd_gamma = (dyf * x_hat).sum(dim=(0,) + reduce_axes)  # 仅保留 C
    pd_beta = dyf.sum(dim=(0,) + reduce_axes)  # 仅保留 C
    return [pd_x, pd_gamma, pd_beta]


def _tp_widen(t):
    """三方腿的计算档: **跟 NPU 的加宽行为**(ttk_golden_logic.md 三/四 的 GPU 列)。

    内核把 f16/bf16 抬到 f32 做中间量, f32 不再加宽; 三方腿必须同步, 出口再窄回。
    不同步的后果是三方被人为劣化 —— 它把整条归约留在 f16 算, 误差远大于内核,
    cross_check 的分母虚高、比值系统性偏小, 内核有缺陷也照样 PASS。
    (L2 实测: fp16 档 mare 比值中位 0.0135, fp32 档 0.4697, 差 35 倍。)
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


class _InstanceNormGradCompose:
    """三方标杆：torch 拼接在远端 GPU 执行，fp32（竞品自然精度，不抬 fp64——
    否则分母趋零、cross_check 比值爆表，会把内核误判成缺陷；三方须同精度对等）。
    参数名与 def.cpp 逐字一致（dy/x/variance/mean/gamma）。
    计算档由 `_tp_widen` 跟随内核(f16/bf16 抬 f32); 出口必须 cast 回 NPU 的输出 dtype
    (= 输入 dtype), 否则竞品留在 fp32 而 NPU 是 fp16 时 ratio 凭空爆表
    (gnsq 实测 mare 961→1.0 的教训)。两步缺一不可: 只窄回不加宽 = 三方被劣化、掩盖缺陷。
    """

    def __init__(self, **_):
        pass

    def __call__(self, dy, x, variance, mean, gamma, **_):
        nd = x.dim()
        C = x.shape[-1]
        reduce_axes = tuple(range(1, nd - 1))
        m = 1
        for ax in reduce_axes:
            m *= x.shape[ax]

        # 计算档跟内核: f16/bf16 抬 f32(内核中间量即 f32), 出口再窄回 dy.dtype。
        dyw, xw = _tp_widen(dy), _tp_widen(x)
        pshape = [x.shape[0]] + [1] * (nd - 2) + [C]
        varb = _tp_widen(variance).reshape(pshape)
        meanb = _tp_widen(mean).reshape(pshape)
        gshape = [1] * nd
        gshape[-1] = C
        gammab = _tp_widen(gamma).reshape(gshape)

        rstd = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -0.5)
        # rstd^3 必须与算子实现逐字一致(A2 tbe impl instance_norm_grad.py:117-118 与 arch35 内核
        # 均为 rstd*rstd*rstd,三次乘法三次舍入)。写成 pow(v,-1.5) 只舍入一次,竞品会凭空比被测
        # 实现准约 2 倍,三方比的就不再是"同一算法下谁实现得更好",而是"用了哪个公式"。
        # fp64 golden 不受影响(两种写法差 ~2e-16),故只在三方 compose 这一处对齐。
        rstd3 = rstd * rstd * rstd
        xc = xw - meanb
        pd_xl = dyw * gammab
        # 空轴集求和取恒等，理由同 _compute（rank == 2 无空间轴）。
        pd_var = _sum_axes(-0.5 * pd_xl * xc * rstd3, reduce_axes)
        pd_mean = _sum_axes(-1.0 * pd_xl * rstd, reduce_axes)
        inv_m = 0.0 if m == 0 else 1.0 / m
        pd_x = pd_xl * rstd + pd_var * (2.0 * inv_m) * xc + pd_mean * inv_m
        x_hat = xc * rstd
        pd_gamma = (dyw * x_hat).sum(dim=(0,) + reduce_axes)
        pd_beta = dyw.sum(dim=(0,) + reduce_axes)
        return [pd_x.to(dy.dtype), pd_gamma.to(dy.dtype), pd_beta.to(dy.dtype)]


def _inject_nonfinite(dy, x, variance, mean, gamma, testcase_name=""):
    """DFX-nonfinite 档的定点注入: 只对用例名含 `_nonfinite` 的用例生效。

    为什么不能靠 CSV 的 input_data_ranges 写 inf/nan: TTK 的 RandomData 会把值域里的
    inf/nan **钳到 dtype 极值**(ttk/utilities/data.py `_digitize_inf_nan`), 写了也造不出
    非有限数据 —— 那一档会变成"名字叫 nonfinite、数据却全是普通数"的空跑。

    只注入**数据面**输入(dy, x), 不注入权重/统计量(variance, mean, gamma):
    后者是逐通道广播量, 注入会让整通道输出非有限, 掩盖"非有限值沿计算链如何传播"这一档
    真正要看的东西。位置固定不随机(随机会让复现依赖 seed, 小 shape 时还可能一个都注不进去):
    首元素 +inf、第 2 个 -inf、第 3 个 nan。元素数 < 3 不注入。

    预期行为: 按 IEEE 语义自然传播, 仍走正常精度判据, 不是拒收档。
    """
    if "_nonfinite" not in (testcase_name or ""):
        return dy, x, variance, mean, gamma
    _t0 = np.ascontiguousarray(dy).copy() if dy is not None else None
    _t1 = np.ascontiguousarray(x).copy() if x is not None else None
    _t2 = np.ascontiguousarray(variance).copy() if variance is not None else None
    _t3 = np.ascontiguousarray(mean).copy() if mean is not None else None
    _t4 = np.ascontiguousarray(gamma).copy() if gamma is not None else None
    if _t0 is not None and _t0.size >= 3:
        _f = _t0.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    if _t1 is not None and _t1.size >= 3:
        _f = _t1.reshape(-1)
        _f[0], _f[1], _f[2] = np.inf, -np.inf, np.nan
    return _t0, _t1, _t2, _t3, _t4


class InstanceNormGradSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray]（舍回输入 dtype）。"""

    def golden(dy, x, variance, mean, gamma, **kwargs):
        ori_dtype = np.asarray(dy).dtype
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(dy)),
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(variance)),
            torch.from_numpy(np.ascontiguousarray(mean)),
            torch.from_numpy(np.ascontiguousarray(gamma)),
            **kwargs,
        )
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        return [
            o.numpy().astype(od[i] if i < len(od) else ori_dtype, copy=False)
            for i, o in enumerate(outs)
        ]

    def customize_inputs(dy, x, variance, mean, gamma, **kwargs):
        return _inject_nonfinite(
            dy, x, variance, mean, gamma, kwargs.get("testcase_name", "")
        )

    third_party = {"torch": _InstanceNormGradCompose}
    tolerance = _TOL


def instance_norm_grad_golden(dy, x, variance, mean, gamma, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(InstanceNormGradSpec.golden(dy, x, variance, mean, gamma, **kwargs))


__spec__ = {"instance_norm_grad": "InstanceNormGradSpec"}
__golden__ = {"kernel": {"instance_norm_grad": "instance_norm_grad_golden"}}

# 【不存在】aclnn 通路：canndev 无 op_api 侧 aclnnInstanceNormGrad，本算子是 TBE-DSL、仅 GE 通路（01 §3.3）。
# 【不存在】e2e 通路：torch_npu 二进制 0 引用（strings libtorch_npu.so 无任何 InstanceNorm 串）；
#   torch 的 InstanceNorm 反向经 batch_norm 分解落到 BatchNormGrad，不调本算子（01 §3.3）。
# 【不存在】onnx / caffe 通路：无对应插件（01 §3.3）。tf 通路存在（tf_plugin 注册 + scope 融合
#   pass），其验证走 aclgrphParseTensorFlow 预生成 .pb，不在 TTK invoke_path 格式内（01 §3.3）。
