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
故 golden 必须是 fp64 真值（竞品算子拼接，非 numpy 纯公式，红线 R3）。
与 instance_norm_grad 的 fp64 golden 约定一致。

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


def _compute(dy, x, variance, mean, **_):
    """torch.Tensor 进 / 出（fp64 真值），返回 [res_gamma, res_beta]，顺序照 def.cpp。"""
    dy_t = dy.to(torch.float64)
    x_t = x.to(torch.float64)
    var_t = variance.to(torch.float64)
    mean_t = mean.to(torch.float64)

    rstd = torch.rsqrt(var_t + _EPS)
    x_norm = (x_t - mean_t) * rstd
    res_gamma = (dy_t * x_norm).sum(dim=_REDUCE_AXES, keepdim=True)
    res_beta = dy_t.sum(dim=_REDUCE_AXES, keepdim=True)
    return [res_gamma, res_beta]


class _IntugCompose:
    """三方标杆：torch 拼接在远端 GPU 执行，fp32（竞品自然精度，不抬 fp64）。

    本算子内核是 fp32+Kahan，比 fp32 朴素竞品更准——cross_check 比值 <1 属预期
    （比值小表示 NPU 误差更小，PASS）；若竞品抬 fp64 则分母趋零、比值爆表，
    会把更准的内核误判成缺陷（三方须同精度对等）。输出与 NPU 同 fp32，无需额外 cast。
    """

    def __init__(self, **_):
        pass

    def __call__(self, dy, x, variance, mean, **_):
        rstd = torch.rsqrt(variance + _EPS)
        x_norm = (x - mean) * rstd
        res_gamma = (dy * x_norm).sum(dim=_REDUCE_AXES, keepdim=True)
        res_beta = dy.sum(dim=_REDUCE_AXES, keepdim=True)
        return [res_gamma, res_beta]


class InTrainingUpdateGradSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray](fp64 真值)。"""

    def golden(dy, x, variance, mean, **kwargs):
        outs = _compute(
            torch.from_numpy(np.ascontiguousarray(dy)),
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(variance)),
            torch.from_numpy(np.ascontiguousarray(mean)),
            **kwargs,
        )
        return [o.numpy().astype(np.float64) for o in outs]

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
