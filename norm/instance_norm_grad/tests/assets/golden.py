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
全部 torch 算子拼接（非 numpy 纯公式，红线 R3），fp64 真值（大规约对消敏感，
fp32 golden 自带误差会误flag内核——内核为 fp32+Kahan，已达 fp32 地板）。

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


def _compute(dy, x, variance, mean, gamma, **_):
    """torch.Tensor 进 / 出（fp64 真值），返回 [pd_x, pd_gamma, pd_beta]，顺序照 def.cpp。"""
    nd = x.dim()
    C = x.shape[-1]
    reduce_axes = tuple(range(1, nd - 1))  # 空间轴 (D,H,W)
    m = 1
    for ax in reduce_axes:
        m *= x.shape[ax]

    dyf = dy.to(torch.float64)
    xf = x.to(torch.float64)

    pshape = [x.shape[0]] + [1] * (nd - 2) + [C]  # [N,1,...,1,C]
    varb = variance.to(torch.float64).reshape(pshape)
    meanb = mean.to(torch.float64).reshape(pshape)
    gshape = [1] * nd
    gshape[-1] = C
    gammab = gamma.to(torch.float64).reshape(gshape)

    rstd = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -0.5)
    rstd3 = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -1.5)

    xc = xf - meanb
    pd_xl = dyf * gammab
    # 必须用 torch 自身的规约:np.sum 会把 torch 张量转成 numpy f64 数组,后续 pd_x 也退化成
    # numpy f64,TTK 据此把图里的 dy 建成 DT_DOUBLE -> EZ3002 算子不支持。
    pd_var = (-0.5 * pd_xl * xc * rstd3).sum(dim=reduce_axes, keepdim=True)
    pd_mean = (-1.0 * pd_xl * rstd).sum(dim=reduce_axes, keepdim=True)
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


class _InstanceNormGradCompose:
    """三方标杆：torch 拼接在远端 GPU 执行，fp32（竞品自然精度，不抬 fp64——
    否则分母趋零、cross_check 比值爆表，会把内核误判成缺陷；三方须同精度对等）。
    参数名与 def.cpp 逐字一致（dy/x/variance/mean/gamma）。
    ⚠️ 输出必须 cast 回 NPU 的输出 dtype（= 输入 dtype），否则竞品留在 fp32 而
    NPU 是 fp16 时 ratio 凭空爆表（gnsq 实测 mare 961→1.0 的教训）。
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

        pshape = [x.shape[0]] + [1] * (nd - 2) + [C]
        varb = variance.reshape(pshape)
        meanb = mean.reshape(pshape)
        gshape = [1] * nd
        gshape[-1] = C
        gammab = gamma.reshape(gshape)

        rstd = torch.pow(varb + INSTANCE_NORM_GRAD_EPS, -0.5)
        # rstd^3 必须与算子实现逐字一致(A2 tbe impl instance_norm_grad.py:117-118 与 arch35 内核
        # 均为 rstd*rstd*rstd,三次乘法三次舍入)。写成 pow(v,-1.5) 只舍入一次,竞品会凭空比被测
        # 实现准约 2 倍,三方比的就不再是"同一算法下谁实现得更好",而是"用了哪个公式"。
        # fp64 golden 不受影响(两种写法差 ~2e-16),故只在三方 compose 这一处对齐。
        rstd3 = rstd * rstd * rstd
        xc = x - meanb
        pd_xl = dy * gammab
        pd_var = (-0.5 * pd_xl * xc * rstd3).sum(dim=reduce_axes, keepdim=True)
        pd_mean = (-1.0 * pd_xl * rstd).sum(dim=reduce_axes, keepdim=True)
        pd_x = pd_xl * rstd + pd_var * (2.0 / m) * xc + pd_mean * (1.0 / m)
        x_hat = xc * rstd
        pd_gamma = (dy * x_hat).sum(dim=(0,) + reduce_axes)
        pd_beta = dy.sum(dim=(0,) + reduce_axes)
        return [pd_x.to(dy.dtype), pd_gamma.to(dy.dtype), pd_beta.to(dy.dtype)]


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
