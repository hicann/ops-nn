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
TTK TestSpec for mse_loss_v2 (kernel / GEIR / aclnn / e2e 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，竞品接口 F.mse_loss（非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一竞品接口在远端 GPU 上执行（cross_check 比值的另一腿）
    tolerance    —— 浮点输出 cross_check（NPU/竞品 相对 golden 的误差比值）

Canonical IO order (mse_loss_v2_def.cpp):
    inputs : input, target
    outputs: output
    attrs  : reduction(OPTIONAL str="mean", in {none, sum, mean})

kernel 内部 fp16/bf16->fp32 计算再 RNE 舍回，golden 同样抬 fp32 计算、末位照 output_dtypes
舍回（中间精度与算子实现一致，不抬 fp64）。

aclnn 通路复用 loss/mse_loss 的 aclnnMseLoss（soc∈{910b,910_93,310p} 且 self.shape==target.shape
且 ND/NCL 时 dispatch 到 MSELossV2）：参数名取 aclnn_mse_loss.h（self/target），reduction 为
int64（0=none/1=mean/2=sum，aclnn_mse_loss.cpp REDUCTION_*_NUM），golden 内映射回字符串。
e2e 通路即 torch.nn.functional.mse_loss（torch_npu 经 PrivateUse1 落到 aclnnMseLoss）。
"""

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None

# Spec.tolerance 只认官方四标准：stat_rel_err / binary_equal / cross_check / quant
# （close、requant 是 CLI 专用别名，写进 Spec 会 InvalidSpecError）。
_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}

_REDUCTION_NUM2STR = {0: "none", 1: "mean", 2: "sum"}


def _attr(kwargs, name, default):
    """attributes 可能平铺在 kwargs，也可能收在 kwargs['attributes'] dict；字符串做类型归一。"""
    v = kwargs.get(name)
    if v is None:
        attrs = kwargs.get("attributes")
        if isinstance(attrs, dict):
            v = attrs.get(name)
    if v is None:
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "false", "yes", "no", "1", "0") and isinstance(default, bool):
            return s in ("true", "yes", "1")
        try:
            return type(default)(v)
        except Exception:
            return default
    return v


def _reduction_str(v):
    """reduction 归一到字符串：kernel/e2e 通路本来就是 str，aclnn 通路是 int64。"""
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_NUM2STR[int(v)]


def _f32_floor(t):
    """fp16/bf16 抬 fp32（CPU half 支持残缺，且与 NPU 内核 fp32 计算一致）；
    fp32/fp64 照单全收——cross_check 场景框架按 golden_mode=Promote 自动把输入抬一档
    （fp32→fp64），golden 不自行 cast、不替框架做精度决策（砍回 fp32 会废掉 Promote）。
    """
    return t.to(torch.float32) if t.dtype in (torch.float16, torch.bfloat16) else t


def _compute(input, target, **kwargs):
    """torch.Tensor 进 / 出，返回 list[Tensor]（舍回输出 dtype 由各路壳负责）。"""
    reduction = _reduction_str(_attr(kwargs, "reduction", "mean"))
    y = F.mse_loss(_f32_floor(input), _f32_floor(target), reduction=reduction)
    return [y]


def _to_torch(arr):
    """numpy → torch，原样保留 dtype（fp64/fp32 直转；fp16 直转；
    bf16 经 ml_dtypes 按位 view，无损机械转换，非精度决策）。"""
    a = np.ascontiguousarray(arr)
    if _bf16 is not None and a.dtype == _bf16:
        return torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(a)


class _MseLossCompose:
    """三方标杆：竞品接口 F.mse_loss 直出，由 TTK 派发到远端 GPU 执行。

    参数绑定契约：属性喂 __init__、输入喂 __call__，参数名与 def.cpp 逐字一致
    （input / target / reduction）。输出 dtype 天然随输入（= NPU 输出 dtype），无需额外 cast。
    """

    def __init__(self, *, reduction="mean", **_):
        self.reduction = _reduction_str(reduction)

    def __call__(self, input, target, **_):
        return [F.mse_loss(input, target, reduction=self.reduction)]


class _MseLossAclnnCompose:
    """aclnn 通路三方标杆（aclnnMseLoss）：输入名取 aclnn 头文件（self/target）、
    reduction 为 int64(0=none/1=mean/2=sum)。

    ⚠️ `self` 仅位置参数：aclnn 首参名就叫 self，服务端按名绑定时 self=<tensor>
    会以关键字进 **_；若方法的 self 是常规位置关键字参数，调用即撞名
    "got multiple values for argument 'self'"。positional-only 声明后，
    关键字 self 只能落进 **kw，不撞实例参数。__init__/__call__ 都同理要加。
    """

    def __init__(self, /, reduction=1, **_):
        self._red = _reduction_str(reduction)

    def __call__(self, /, **kw):
        x = kw.get("self", kw.get("input"))
        return [F.mse_loss(x, kw["target"], reduction=self._red)]


class MseLossV2KernelSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 list[np.ndarray]。"""

    def golden(input, target, **kwargs):
        outs = _compute(_to_torch(input), _to_torch(target), **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        target_dt = od[0] if od else str(np.asarray(input).dtype)
        out = outs[0].detach().cpu().numpy()
        if target_dt == "bfloat16":
            out = out.astype(np.float32).astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target_dt)
        return [out]

    third_party = {"torch": _MseLossCompose}
    tolerance = _TOL


class MseLossV2AclnnSpec:
    """aclnn 通路 spec：golden 收已 H2D 的 torch.Tensor、返 Tensor。

    参数名取 op_api/aclnn_mse_loss.h（self/target），reduction 为 int64（0=none/1=mean/2=sum）。
    首参名 self 照写：TTK 用 getattr(cls,"golden") 从类取出，普通函数不绑定实例。
    """

    def golden(self, target, reduction=1, out=None, **kwargs):
        # 签名与 aclnnMseLossGetWorkspaceSize 一致(不含 workspaceSize/executor);
        # reduction 为 int64(0=none/1=mean/2=sum),golden 内映射回字符串。
        y = _compute(self, target, reduction=reduction, **kwargs)[0]
        return [y.to(self.dtype)]

    # TTK ≥193da3e 起 aclnn 通路支持三方;指向 aclnn 名参数版 compose(首参 self 防撞名)。
    third_party = {"torch": _MseLossAclnnCompose}
    tolerance = _TOL


class MseLossV2TorchSpec:
    """e2e 通路 spec（torch.nn.functional.mse_loss）：golden 收已 H2D 的 torch.Tensor。

    ⚠️ 签名按 torch API 全参数序：框架按 param plan **位置**传全量参数
    （input, target, size_average, reduce, reduction），少声明会报"6 were given"。
    """

    def golden(input, target, *args, **kwargs):
        # 框架按 param plan 位置传全量参数(input, target, out, size_average, reduce,
        # reduction —— docstring 解析含 out,共 6 个)。稳健取 reduction:位置参数里的
        # 合法字符串,或 kwargs;缺省 mean。
        reduction = kwargs.get("reduction")
        if reduction is None:
            strs = [
                a for a in args if isinstance(a, str) and a in ("none", "mean", "sum")
            ]
            reduction = strs[0] if strs else "mean"
        y = _compute(input, target, reduction=reduction, **kwargs)[0]
        return [y.to(input.dtype)]

    third_party = {"torch": _MseLossCompose}  # 【预留】同 AclnnSpec，当前不被取用
    tolerance = _TOL


def mse_loss_v2_golden(input, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return MseLossV2KernelSpec.golden(input, target, **kwargs)


__spec__ = {
    "mse_loss_v2": "MseLossV2KernelSpec",
    "aclnnMseLoss": "MseLossV2AclnnSpec",
    "torch.nn.functional.mse_loss": "MseLossV2TorchSpec",
}
__golden__ = {"kernel": {"mse_loss_v2": "mse_loss_v2_golden"}}

# 【不存在】tf / onnx 通路：无 framework 插件（01_requirement.md §3.3）。
# 注：torch 图模式经 GE 图通路到达，不单独注册（同 geir）。
