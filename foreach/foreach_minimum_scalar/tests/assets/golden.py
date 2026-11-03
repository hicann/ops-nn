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
foreach_minimum_scalar 两方 Golden（无 GPU 版）

由用户 golden 适配而来，用于"NPU vs CPU golden"两方精度比对：
来源: /data/w50064389/SIMT-LongTails/ForeachMinimumScalar/deliverables/thirdparty_golden/golden.py

适配内容（相对原文件）：
1. tolerance: cross_check/L1 → mix_tolerance（生态混合容差标准；无需三方参考即可出 PASS/FAIL）
   保留整型 binary_equal
2. kernel golden 增加 bf16 (ml_dtypes 载体 / numpy 'V' dtype) 处理：
   view 还原 → fp32 计算 → 输出按用例 output_dtypes 转回；修复原版 bf16 TypeError
3. 补充 aclnnForeachMinimumScalarV2 注册（原版只注册 V1）
4. third_party 保留并按名绑定类形式声明（Kernel/ACLNN/E2E 三通路均有 torch 第三方参考；
   两方 mix_tolerance 不消费 third_party，仅在具备 XPU endpoint 的 cross_check 轮次中启用）
5. E2E 参考增加超范围标量回退：bf16/低精度张量 + 超出该精度的标量时 torch CPU 直接调用会
   RuntimeError（a5-torch-route L0_019 已复现），回退为 fp32 上采样计算后按原 dtype 回写，
   与 kernel 的 fp32 计算路径一致；正常可调用场景行为不变

公式: y[i] = minimum(x[i], scalar)
"""

__spec__ = {
    "foreach_minimum_scalar": "ForeachMinimumScalarKernelSpec",
    "aclnnForeachMinimumScalar": "AclnnForeachMinimumScalarSpec",
    "aclnnForeachMinimumScalarV2": "AclnnForeachMinimumScalarSpec",
    "torch._foreach_minimum": "TorchForeachMinimumScalarSpec",
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


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item()
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item()
    return value


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
    return torch.from_numpy(np.ascontiguousarray(array))


def _target_dtype(entry):
    if entry is None:
        return None
    return entry[0] if isinstance(entry, (tuple, list)) else str(entry)


def _cast_out(tensor, target):
    if target == "bfloat16":
        return tensor.numpy().astype(_bf16) if _bf16 is not None else tensor.numpy()
    return tensor.numpy().astype(target)


def _foreach_minimum_safe(x, scalar_value):
    """E2E 参考用的稳健调用：低精度张量叠加超范围标量时（torch CPU 直接调用抛 RuntimeError）
    回退为 f32 标量取最近可表示值 + fp32 上采样计算并按原 dtype 回写；正常可调用场景行为不变。"""
    try:
        return list(torch._foreach_minimum(x, scalar_value))
    except RuntimeError:
        scalar_f32 = float(np.float32(scalar_value))
        results = []
        for tensor in x:
            if tensor.dtype == torch.float32:
                result = torch._foreach_minimum([tensor], scalar_f32)[0]
            else:
                source_dtype = tensor.dtype
                result = torch._foreach_minimum([tensor.to(torch.float32)], scalar_f32)[
                    0
                ].to(source_dtype)
            results.append(result)
        return results


class _TorchForeachMinimumScalarImpl:
    """Kernel/ACLNN 通路三方参考：按 TTK 具名绑定（__init__ 取 scalar，__call__ 取 x）。"""

    def __init__(self, scalar, **kwargs):
        self._scalar = _to_scalar(scalar)

    def __call__(self, x, **kwargs):
        return list(torch._foreach_minimum(x, self._scalar))


class _TorchForeachMinimumScalarE2EImpl:
    """E2E 通路三方参考：*args 按池顺序取首参（torch API 首参名为 self），scalar 按名绑定。"""

    def __init__(self, scalar, **kwargs):
        self._scalar = _to_scalar(scalar)

    def __call__(self, *args, scalar=None, **kwargs):
        return _foreach_minimum_safe(args[0], self._scalar)


_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarImpl}
_E2E_THIRD_PARTY = {"torch": _TorchForeachMinimumScalarE2EImpl}


class ForeachMinimumScalarKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。y[i] = minimum(x[i], scalar)"""

    def golden(x, scalar, **kwargs):
        output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
        s = _to_scalar(scalar)
        results = []
        for index, a in enumerate(x):
            ta = _to_torch_tensor(a)
            out = torch._foreach_minimum([ta], s)[0]
            target = None
            if output_dtypes is not None and index < len(output_dtypes):
                target = _target_dtype(output_dtypes[index])
            if target is None:
                target = str(np.asarray(a).dtype)
            results.append(_cast_out(out, target))
        return results

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class AclnnForeachMinimumScalarSpec:
    """ACLNN 路径（V1/V2）— list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x, scalar, out=None, **kwargs):
        s = _to_scalar(scalar)
        result = list(torch._foreach_minimum(x, s))
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return list(result)

    third_party = _THIRD_PARTY
    tolerance = _TOLERANCE


class TorchForeachMinimumScalarSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x=None, scalar=None, self=None, **kwargs):
        actual_x = self if self is not None else x
        s = _to_scalar(scalar)
        return _foreach_minimum_safe(actual_x, s)

    third_party = _E2E_THIRD_PARTY
    tolerance = _TOLERANCE
