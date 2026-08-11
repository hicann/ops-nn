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
TTK TestSpec for multilabel_margin_loss (kernel / GEIR / aclnn / e2e 通路, arch35/Ascend950).

三份资产各司其职：
    golden       —— 真值，竞品接口 torch.ops.aten.multilabel_margin_loss_forward
                   （aten forward 即权威参照，非 numpy 纯公式，红线 R3）
    third_party  —— 三方标杆，同一竞品 aten 接口在远端 GPU 上执行
    tolerance    —— 浮点输出 y 走 cross_check；is_target(int32) 承载掩码语义，
                   差 1 就是错，保持 binary_equal（不套 ±1 容忍）

Canonical IO order (multilabel_margin_loss_def.cpp):
    inputs : x(fp16/bf16/fp32), target(int32)
    outputs: y(同 x dtype), is_target(int32)
    attrs  : reduction(OPTIONAL str="mean", in {none, sum, mean})

aclnn 通路参数名取 aclnn_multilabel_margin_loss.h（self/target），reduction 为 int64
（0=none/1=mean/2=sum），golden 内映射回字符串。e2e 通路 torch.nn.functional.multilabel_margin_loss
只返回 loss（is_target 是 NPU 内部输出，e2e 不暴露），故 TorchSpec 只产 y。

aten forward 要求 target 为 int64（LongTensor），golden/三方统一由 int32 转 int64 再调；
fp16/bf16 在 CPU 侧 aten 不支持，抬 fp32 计算再舍回（与 NPU 内核 fp32 累加一致，不抬 fp64）。
"""

import numpy as np
import torch

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
    "int32": {"standard": "binary_equal"},
}

_REDUCTION_STR2INT = {"none": 0, "mean": 1, "sum": 2}
_REDUCTION_INT2STR = {0: "none", 1: "mean", 2: "sum"}


def _attr(kwargs, name, default):
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
    if isinstance(v, str):
        return v.strip().lower()
    return _REDUCTION_INT2STR[int(v)]


def _compute(x, target, **kwargs):
    """torch.Tensor 进 / 出，返回 [y, is_target]，顺序照 def.cpp 输出序。

    fp16/bf16 输入抬 fp32 计算（aten CPU 不支持 half；与 NPU 内核 fp32 累加一致）。
    返回的 y 保持计算精度，舍回 x dtype 由各路壳负责。
    """
    reduction = _reduction_str(_attr(kwargs, "reduction", "mean"))
    low_prec = x.dtype in (torch.float16, torch.bfloat16)
    xf = x.to(torch.float32) if low_prec else x
    out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
        xf, target.to(torch.int64), _REDUCTION_STR2INT[reduction]
    )
    return [out, is_target]


def _to_torch(a):
    """numpy → torch，原样保留 dtype；bf16(ml_dtypes) 按位 view（无损机械转换）。"""
    arr = np.ascontiguousarray(a)
    if _bf16 is not None and arr.dtype == _bf16:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _cast_np(t, target_dt):
    """torch 计算结果 → numpy 目标 dtype（bf16 经 ml_dtypes）。"""
    arr = (
        t.detach().cpu().to(torch.float32).numpy()
        if t.dtype == torch.bfloat16
        else t.detach().cpu().numpy()
    )
    if target_dt == "bfloat16":
        return arr.astype(_bf16) if _bf16 is not None else arr
    return arr.astype(target_dt)


class _MmlCompose:
    """三方标杆：aten forward 在远端 GPU 执行。参数名与 def.cpp 逐字一致（x/target/reduction）。

    GPU 上抬 fp32 计算再舍回 x dtype（half CUDA 支持度不一，且竞品留在 fp32 会让
    cross_check ratio 凭空爆表——输出必须 cast 回 NPU 的输出 dtype）。
    """

    def __init__(self, *, reduction="mean", **_):
        self.reduction = _reduction_str(reduction)

    def __call__(self, x, target, **_):
        out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
            x.to(torch.float32),
            target.to(torch.int64),
            _REDUCTION_STR2INT[self.reduction],
        )
        return [out.to(x.dtype), is_target.to(torch.int32)]


class _MmlAclnnCompose:
    """aclnn 通路三方标杆（aclnnMultilabelMarginLoss）：输入名取 aclnn 头文件
    （self/target），reduction 为 int64(0=none/1=mean/2=sum)；is_target 跟随 self dtype
    （A5 aclnn 契约）。`self` 仅位置参数防撞名（acl nn 首参名即 self，
    服务端按名绑定时关键字 self 会撞实例参数，positional-only 后落入 **kw）。"""

    def __init__(self, /, reduction=1, **_):
        self._red = _reduction_str(reduction)

    def __call__(self, /, **kw):
        x = kw.get("self", kw.get("input"))
        out, is_target = torch.ops.aten.multilabel_margin_loss_forward(
            x.to(torch.float32),
            kw["target"].to(torch.int64),
            _REDUCTION_STR2INT[self._red],
        )
        return [out.to(x.dtype), is_target.to(x.dtype)]


class _MmlTorchCompose:
    """e2e 通路三方标杆（torch.nn.functional.multilabel_margin_loss）：输入名取 torch API
    签名（input/target）。e2e 只产 y（API 不暴露 is_target）。"""

    def __init__(self, *, reduction="mean", **_):
        self._red = _reduction_str(reduction)

    def __call__(self, input, target, **_):
        out, _is_target = torch.ops.aten.multilabel_margin_loss_forward(
            input.to(torch.float32),
            target.to(torch.int64),
            _REDUCTION_STR2INT[self._red],
        )
        return [out.to(input.dtype)]


class MultilabelMarginLossKernelSpec:
    """kernel / GEIR 通路 spec：golden 收 numpy.ndarray、返 [y, is_target]。"""

    def golden(x, target, **kwargs):
        outs = _compute(_to_torch(x), _to_torch(target), **kwargs)
        od = kwargs.get("output_dtypes") or []
        od = [d[0] if isinstance(d, (list, tuple)) else str(d) for d in od]
        y_dt = od[0] if len(od) > 0 else str(np.asarray(x).dtype)
        t_dt = od[1] if len(od) > 1 else "int32"
        return [_cast_np(outs[0], y_dt), _cast_np(outs[1], t_dt)]

    third_party = {"torch": _MmlCompose}
    tolerance = _TOL


class MultilabelMarginLossAclnnSpec:
    """aclnn 通路 spec：golden 收已 H2D 的 torch.Tensor。

    参数名取 aclnn_multilabel_margin_loss.h（self/target），reduction 为 int64
    （0=none/1=mean/2=sum）。首参名 self 照写：TTK 从类取普通函数，不绑定实例。
    """

    def golden(self, target, reduction=1, out=None, isTarget=None, **kwargs):
        # 签名与 aclnnMultilabelMarginLossGetWorkspaceSize 一致(不含 workspaceSize/executor);
        # reduction 为 int64(0=none/1=mean/2=sum)。
        outs = _compute(self, target, reduction=reduction, **kwargs)
        # 🆕 A5 aclnn 契约：is_target 跟随 self dtype(torch 契约 is_target==self，免 int32→float
        # Cast 依赖)；GE 路径保 int32 对齐 A2。见 design_decision_is_target_dtype.md。
        return [outs[0].to(self.dtype), outs[1].to(self.dtype)]

    # TTK ≥193da3e 起 aclnn 通路支持三方;指向 aclnn 名参数版 compose(首参 self 防撞名)。
    third_party = {"torch": _MmlAclnnCompose}
    tolerance = _TOL


class MultilabelMarginLossTorchSpec:
    """e2e 通路 spec（torch.nn.functional.multilabel_margin_loss）：只产 y（API 不暴露 is_target）。

    ⚠️ 签名按 torch API 全参数序：框架按 param plan **位置**传全量参数
    （input, target, size_average, reduce, reduction），少声明会报参数数错。
    """

    def golden(input, target, *args, **kwargs):
        # 框架按 param plan 位置传全量参数(docstring 解析含 out,共 6 个)。
        # 稳健取 reduction:位置参数里的合法字符串,或 kwargs;缺省 mean。
        reduction = kwargs.get("reduction")
        if reduction is None:
            strs = [
                a for a in args if isinstance(a, str) and a in ("none", "mean", "sum")
            ]
            reduction = strs[0] if strs else "mean"
        outs = _compute(input, target, reduction=reduction, **kwargs)
        return [outs[0].to(input.dtype)]

    third_party = {
        "torch": _MmlTorchCompose
    }  # e2e 名(input/target),与 kernel/aclnn 版区分
    tolerance = _TOL


def multilabel_margin_loss_golden(x, target, **kwargs):
    """保留 __golden__ 约定入口（上库件，签名照 def.cpp），与 Spec 共用同一实现。"""
    return tuple(MultilabelMarginLossKernelSpec.golden(x, target, **kwargs))


__spec__ = {
    "multilabel_margin_loss": "MultilabelMarginLossKernelSpec",
    "aclnnMultilabelMarginLoss": "MultilabelMarginLossAclnnSpec",
    "torch.nn.functional.multilabel_margin_loss": "MultilabelMarginLossTorchSpec",
}
__golden__ = {"kernel": {"multilabel_margin_loss": "multilabel_margin_loss_golden"}}

# 【不存在】tf / onnx 通路：tf_plugin 无本体（仅 shape 变体）、onnx 无（01_requirement.md §3.3）。
# 注：torch 图模式经 GE 图通路到达，不单独注册（同 geir）。
