#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
from __future__ import annotations

import importlib.util as _ilu
import pathlib as _pathlib

import numpy as np

try:
    import torch
except Exception:
    torch = None


__golden__ = {
    "kernel": {
        "ConfusionSoftmaxGrad": "confusion_softmax_grad_golden",
        "confusion_softmax_grad": "confusion_softmax_grad_golden",
    },
}


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        if value.dtype is torch.bfloat16:
            return value.to(torch.float32).cpu().numpy(), "bfloat16"
        return value.detach().cpu().numpy(), str(value.dtype).replace("torch.", "")
    array = np.asarray(value)
    return array, getattr(array.dtype, "name", str(array.dtype))


def _round_like(value, dtype_name):
    if dtype_name == "float16":
        return np.asarray(value, dtype=np.float16)
    if dtype_name == "bfloat16" and torch is not None:
        return (
            torch.from_numpy(np.asarray(value, dtype=np.float32))
            .to(torch.bfloat16)
            .to(torch.float32)
            .cpu()
            .numpy()
        )
    return np.asarray(value, dtype=np.float32)


def confusion_softmax_grad_golden(grad, x, **attrs):
    del attrs
    grad_np, dtype_name = _to_numpy(grad)
    x_np, _ = _to_numpy(x)
    gf = np.asarray(grad_np, dtype=np.float32)
    xf = np.asarray(x_np, dtype=np.float32)
    summed = np.sum(gf * xf, axis=-1, keepdims=True)
    return _round_like(gf - summed, dtype_name)


def reference(inputs: dict, attrs: dict) -> dict:
    ordered = [inputs[key] for key in sorted(inputs) if key.startswith("input_")]
    result = compute(*ordered, **dict(attrs or {}))
    return {"output_0": result}


# ---------------------------------------------------------------------------
# GEIR/TTK golden 接口（自包含单文件约定，供 run_*_geir.py --golden-adapter 与
# TTK golden plugin 调用）。共用逻辑集中在 op_ref/_golden_utils.py（单一共享文件，
# 避免散弹修改）；本段仅为薄委托。
# ---------------------------------------------------------------------------


def _load_golden_utils():
    """动态加载 op_ref/_golden_utils.py（golden 常被 spec_from_file_location 无包上下文加载）。"""
    _here = _pathlib.Path(__file__).resolve()
    for _parent in _here.parents:
        for _cand in (
            _parent / "_golden_utils.py",
            _parent / "op_ref" / "_golden_utils.py",
        ):
            if _cand.is_file():
                _spec = _ilu.spec_from_file_location("_golden_utils", _cand)
                if _spec and _spec.loader:
                    _mod = _ilu.module_from_spec(_spec)
                    _spec.loader.exec_module(_mod)
                    return _mod
    return None


_utils = _load_golden_utils()
_OP_NORM = "confusionsoftmaxgrad"


def call_registered(level, api_name, *inputs, **attrs):
    if _utils is not None:
        return _utils.call_registered_with_globals(
            __golden__, globals(), level, api_name, _OP_NORM, *inputs, **attrs
        )
    funcs = __golden__.get(str(level), {})
    func_name = funcs.get(api_name) if isinstance(funcs, dict) else None
    func = globals().get(str(func_name)) if func_name else None
    if callable(func):
        return func(*inputs, **attrs)
    raise NotImplementedError(f"No golden for level={level} api={api_name}")


def call_input(api_name, *inputs, **attrs):
    if _utils is not None:
        return _utils.call_input_with_globals(
            globals().get("__input__", {}),
            globals(),
            api_name,
            _OP_NORM,
            *inputs,
            **attrs,
        )
    return None


def compute(*inputs, **attrs):
    if _utils is not None:
        return _utils.compute_from_registry(
            __golden__, globals(), _OP_NORM, *inputs, **attrs
        )
    for _level in ("kernel", "aclnn", "e2e"):
        _funcs = __golden__.get(_level, {})
        if isinstance(_funcs, dict):
            for _fn in _funcs.values():
                _f = globals().get(str(_fn))
                if callable(_f):
                    return _f(*inputs, **attrs)
    raise NotImplementedError("No compute function found")
