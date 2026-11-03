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
"""
foreach_sub_scalar_list 两方 Golden（无 GPU 版）

由用户 golden 适配而来，用于"NPU vs CPU golden"两方精度比对：
来源: /data/w50064389/SIMT-LongTails/ForeachSubScalarList/deliverables/thirdparty_golden/golden.py

适配内容（相对原文件）：
1. tolerance: cross_check/L1 → mix_tolerance（生态混合容差标准；无需三方参考即可出 PASS/FAIL）
   保留整型 binary_equal
2. kernel golden 增加 bf16 (ml_dtypes 载体 / numpy 'V' dtype) 处理与按用例 output_dtypes 回转，
   修复原版逐元素相减在 bf16/numpy 载体上的 dtype 问题
3. third_party 保留为按名绑定类形式（两方 mix_tolerance 不消费 third_party，
   仅在具备 XPU endpoint 的 cross_check 轮次中启用）

公式: y[i] = x[i] - scalars[i]
"""

__spec__ = {
    "foreach_sub_scalar_list": "ForeachSubScalarListKernelSpec",
    "ForeachSubScalarList": "ForeachSubScalarListKernelSpec",
    "aclnnForeachSubScalarList": "AclnnForeachSubScalarListSpec",
    "torch._foreach_sub": "TorchForeachSubScalarListSpec",
    "torch._foreach_sub.ScalarList": "TorchForeachSubScalarListSpec",
}

import numpy as np
import torch

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None

_TOLERANCE = {
    "float16": {"standard": "mix_tolerance"},
    "float32": {"standard": "mix_tolerance"},
    "bfloat16": {"standard": "mix_tolerance"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
}


def _as_bf16_view(array):
    if array.dtype.kind == "V" and _bf16 is not None:
        return array.view(_bf16)
    return array


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        return _as_bf16_view(value).reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    return value


def _to_scalars(value):
    if isinstance(value, np.ndarray):
        return _as_bf16_view(value).reshape(-1).tolist()
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).tolist()
    return [_to_scalar(item) for item in value]


def _per_tensor_dtypes(output_dtypes):
    if output_dtypes is None:
        return None
    flat = []
    for entry in output_dtypes:
        if isinstance(entry, (tuple, list)):
            flat.extend(entry)
        else:
            flat.append(entry)
    return flat


def _to_torch_tensor(value):
    array = np.asarray(value)
    if array.dtype.kind == "V" and _bf16 is not None:
        array = array.view(_bf16)
    if _bf16 is not None and array.dtype == _bf16:
        return torch.from_numpy(array.astype(np.float32))
    # rank-0 输入不做连续化：np.ascontiguousarray 会把 rank-0 提升为 (1,)，破坏 shape 契约
    if array.ndim > 0:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array)


def _target_dtype(entry):
    if entry is None:
        return None
    return entry[0] if isinstance(entry, (tuple, list)) else str(entry)


def _cast_out(tensor, target):
    if target == "bfloat16":
        return tensor.numpy().astype(_bf16) if _bf16 is not None else tensor.numpy()
    return tensor.numpy().astype(target)


class _TorchForeachSubScalarListImpl:
    """Kernel/ACLNN 通路三方参考：按 TTK 具名绑定（__init__ 取 scalars，__call__ 取 x）。"""

    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, x, **kwargs):
        return list(torch._foreach_sub(x, self._scalars))


class _TorchForeachSubScalarListE2EImpl:
    """E2E 通路三方参考：*args 按池顺序取首参（torch API 首参名为 self），scalars 按名绑定。"""

    def __init__(self, scalars, **kwargs):
        self._scalars = _to_scalars(scalars)

    def __call__(self, *args, scalars=None, **kwargs):
        return list(torch._foreach_sub(args[0], self._scalars))


_THIRD_PARTY = {"torch": _TorchForeachSubScalarListImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachSubScalarListE2EImpl}


class ForeachSubScalarListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。y[i] = x[i] - scalars[i]"""

    def golden(x, scalars, **kwargs):
        output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
        s_list = _to_scalars(scalars)
        results = []
        for index, a in enumerate(x):
            ta = _to_torch_tensor(a)
            s = s_list[index] if index < len(s_list) else s_list[-1]
            out = torch._foreach_sub([ta], [s])[0]
            target = None
            if output_dtypes is not None and index < len(output_dtypes):
                target = _target_dtype(output_dtypes[index])
            if target is None:
                target = str(np.asarray(a).dtype)
            results.append(_cast_out(out, target))
        return results

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachSubScalarListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalars, out=None, **kwargs):
        s_list = _to_scalars(scalars)
        result = list(torch._foreach_sub(x, s_list))
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachSubScalarListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x=None, scalars=None, self=None, **kwargs):
        actual_x = self if self is not None else x
        s_list = _to_scalars(scalars)
        return list(torch._foreach_sub(actual_x, s_list))

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
