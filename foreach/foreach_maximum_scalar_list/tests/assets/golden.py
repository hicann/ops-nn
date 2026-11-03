#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""foreach_maximum_scalar_list 三方 Golden — 标量列表最大值。"""

__spec__ = {
    "foreach_maximum_scalar_list": "ForeachMaximumScalarListKernelSpec",
    "aclnnForeachMaximumScalarList": "AclnnForeachMaximumScalarListSpec",
    "torch._foreach_maximum": "TorchForeachMaximumScalarListSpec",
    "torch._foreach_maximum.ScalarList": "TorchForeachMaximumScalarListSpec",
}

import numpy as np
import ml_dtypes
import torch

_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
    "bfloat16": {"standard": "binary_equal"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _fix_zero_tie(xs, scalars, results):
    """±0 平局时按 IEEE 754 maximum 归一：异号零取 +0，同号 -0 保留 -0。
    与 kernel SimtMaximum 的 `return a + b` 语义一致；非零/非平局元素不改。"""
    fixed = []
    for x, s, r in zip(xs, scalars, results):
        if not torch.is_floating_point(x) or s != 0:
            fixed.append(r)
            continue
        mask = x == 0
        if not bool(mask.any()):
            fixed.append(r)
            continue
        tie_val = (x.to(torch.float32) + float(s)).to(r.dtype)
        fixed.append(torch.where(mask, tie_val, r))
    return fixed


def _to_scalar(v):
    if isinstance(v, np.ndarray):
        return v.reshape(-1)[0].item()
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item()
    return v


def _to_scalars(v):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return v.reshape(-1).tolist()
    return [_to_scalar(t) for t in v]


def _integer_scalar_to_int64(v):
    """Integer scalar -> int64; None when not representable (negative cases go to the operator)."""
    try:
        iv = int(np.int64(v))
    except (OverflowError, ValueError, TypeError):
        return None
    if iv < -(1 << 63) or iv > (1 << 63) - 1:
        return None
    return iv


def _prepare_compute(xs, scalars):
    """Mirror the kernel dtype policy: integer x computes in int64 and floating x in float32,
    both then cast back to the original x dtype. Return None on unrepresentable integer scalars."""
    calc_xs, calc_scalars, out_dtypes = [], [], []
    for t, s in zip(xs, scalars):
        if torch.is_floating_point(t):
            calc_xs.append(t.to(torch.float32))
            with np.errstate(over="ignore"):
                calc_scalars.append(float(np.float32(s)))
            out_dtypes.append(t.dtype)
            continue
        iv = _integer_scalar_to_int64(s)
        if iv is None:
            return None
        calc_xs.append(t.to(torch.int64))
        calc_scalars.append(iv)
        out_dtypes.append(t.dtype)
    return calc_xs, calc_scalars, out_dtypes


def _compute(xs, scalars):
    """Third-party 计算：与 golden/kernel 同 dtype 策略。
    直接 torch._foreach_maximum(x, scalars) 会按 x 的 dtype 收窄标量，double 标量
    超出 x dtype 范围（如 bfloat16 配 -1.7e308）会抛 overflow。故先按
    _prepare_compute 转换（浮点转 float32、整型转 int64），再计算并 cast 回原 dtype。
    """
    prepared = _prepare_compute(xs, scalars)
    if prepared is None:
        return list(torch._foreach_maximum(xs, scalars))
    calc_xs, calc_scalars, out_dtypes = prepared
    result = torch._foreach_maximum(calc_xs, calc_scalars)
    result = _fix_zero_tie(calc_xs, calc_scalars, result)
    return [r.to(dt) for r, dt in zip(result, out_dtypes)]


class _TorchForeachMaximumScalarListImpl:
    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, x, **kwargs):
        return _compute(list(x), self._scalars)


class _TorchForeachMaximumScalarListE2EImpl:
    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, *args, scalars=None):
        return _compute(list(args[0]), self._scalars)


_THIRD_PARTY = {"torch": _TorchForeachMaximumScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachMaximumScalarListE2EImpl}


class ForeachMaximumScalarListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x, scalars, **kwargs):
        int_mode = np.ascontiguousarray(x[0]).dtype == np.int32
        s_list = _to_scalars(scalars)
        if int_mode:
            s_list = [int(np.int64(v)) for v in s_list]
        else:
            with np.errstate(over="ignore"):
                s_list = [float(np.float32(v)) for v in s_list]

        def _to_torch(t):
            a = np.ascontiguousarray(t)
            if a.dtype == ml_dtypes.bfloat16:
                return (
                    torch.from_numpy(a.view(np.uint16)).view(torch.bfloat16).float(),
                    torch.bfloat16,
                )
            if a.dtype == np.float16:
                return torch.from_numpy(a).float(), torch.float16
            if a.dtype == np.int32:
                return torch.from_numpy(a).to(torch.int64), torch.int32
            return torch.from_numpy(a), torch.from_numpy(a).dtype

        converted = [_to_torch(t) for t in x]
        x_list = [c[0] for c in converted]
        out_dtypes = [c[1] for c in converted]

        result = torch._foreach_maximum(x_list, s_list)
        result = _fix_zero_tie(x_list, s_list, result)

        out = []
        for r, dt in zip(result, out_dtypes):
            if dt == torch.bfloat16:
                out.append(
                    r.to(torch.bfloat16)
                    .view(torch.uint16)
                    .numpy()
                    .view(ml_dtypes.bfloat16)
                )
            else:
                out.append(r.to(dt).numpy())
        return out

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMaximumScalarListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。

    aclnn 签名: AclnnForeachMaximumScalarListGetWorkspaceSize(x, scalars, out, workspaceSize, executor)
    """

    def golden(x, scalars, out=None, **kwargs):
        s_list = _to_scalars(scalars)
        # 负向用例：x 与 scalars 列表长度不一致时不做 golden 计算，交由算子/接口拒绝；
        # 返回占位结果，避免 golden 先行抛错把负向用例误报成 golden 缺陷。
        if len(x) != len(s_list):
            return list(out) if out is not None else [t.clone() for t in x]
        prepared = _prepare_compute(x, s_list)
        # 负向用例：整型标量超出 int64 可表示范围时同样返回占位
        if prepared is None:
            return list(out) if out is not None else [t.clone() for t in x]
        calc_x, calc_s, out_dtypes = prepared
        result = torch._foreach_maximum(calc_x, calc_s)
        result = _fix_zero_tie(calc_x, calc_s, result)
        result = [r.to(dt) for r, dt in zip(result, out_dtypes)]
        if out is not None:
            for o, r in zip(out, result):
                if o.numel() == r.numel():
                    o.copy_(r)
                    continue
                if o.numel() < r.numel():
                    # 负向用例：out 小于 x，期望被接口拒绝；不抛错，保留 out 占位
                    continue
                if not o.is_contiguous():
                    raise RuntimeError(
                        "larger non-contiguous out is not expressible by this golden"
                    )
                o.view(-1)[: r.numel()].copy_(r.reshape(-1))
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMaximumScalarListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalars, **kwargs):
        s_list = _to_scalars(scalars)
        prepared = _prepare_compute(x, s_list)
        if prepared is None:
            return [t.clone() for t in x]
        calc_x, calc_s, out_dtypes = prepared
        result = torch._foreach_maximum(calc_x, calc_s)
        result = _fix_zero_tie(calc_x, calc_s, result)
        return [r.to(dt) for r, dt in zip(result, out_dtypes)]

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
