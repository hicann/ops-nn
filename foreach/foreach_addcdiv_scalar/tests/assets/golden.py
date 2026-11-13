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
foreach_addcdiv_scalar 两方 Golden（无 GPU 版）

由用户 golden 适配而来，用于"NPU vs CPU golden"两方精度比对：
来源: /data/w50064389/SIMT-LongTails/ForeachAddcdivScalar/deliverables/thirdparty_golden/golden.py

适配内容（相对原文件）：
1. tolerance: cross_check/L1 → mix_tolerance（生态混合容差标准；无需三方参考即可出 PASS/FAIL）
   保留整型 binary_equal
2. kernel golden 增加 bf16 (ml_dtypes 载体 / numpy 'V' dtype) 处理：
   view 还原 → fp32 计算 → 输出按用例 output_dtypes 转回；修复原版 bf16 TypeError
3. 补充 aclnnForeachAddcdivScalarV2 注册（原版只注册 V1）
4. 移除 third_party 声明：两方模式不需要，且原绑定方式不符 TTK 具名参数约定

公式: y[i] = x1[i] + scalar * (x2[i] / x3[i])
"""

__spec__ = {
    "foreach_addcdiv_scalar": "ForeachAddcdivScalarKernelSpec",
    "aclnnForeachAddcdivScalar": "AclnnForeachAddcdivScalarSpec",
    "aclnnForeachAddcdivScalarV2": "AclnnForeachAddcdivScalarSpec",
    "torch._foreach_addcdiv": "TorchForeachAddcdivScalarSpec",
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


def _is_f16_tensor(t):
    if isinstance(t, torch.Tensor):
        return t.dtype == torch.float16
    dt = getattr(t, "dtype", None)
    return dt is not None and getattr(dt, "name", "") == "float16"


def _scalar_for_dtype(scalar, tensors):
    """按算子实际 scalar 精度取整（对齐 DtypeScalarToTensor2）：x1=FLOAT16 时 scalar 按 FLOAT16
    参与计算；x1=FLOAT32/BFLOAT16 时按 FLOAT32。避免 golden 用全精度 scalar 与算子结果差 1 ULP。"""
    s = float(_to_scalar(scalar))
    if tensors and _is_f16_tensor(tensors[0]):
        return float(np.float16(s))
    return float(np.float32(s))


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


class ForeachAddcdivScalarKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。"""

    def golden(x1, x2, x3, scalar, **kwargs):
        output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
        s = _scalar_for_dtype(scalar, x1)
        results = []
        for index, (a, b, c) in enumerate(zip(x1, x2, x3)):
            ta = _to_torch_tensor(a)
            tb = _to_torch_tensor(b)
            tc = _to_torch_tensor(c)
            out = torch._foreach_addcdiv([ta], [tb], [tc], s)[0]
            target = None
            if output_dtypes is not None and index < len(output_dtypes):
                target = _target_dtype(output_dtypes[index])
            if target is None:
                target = str(np.asarray(a).dtype)
            results.append(_cast_out(out, target))
        return results

    tolerance = _TOLERANCE


class AclnnForeachAddcdivScalarSpec:
    """ACLNN 路径（V1/V2）— list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, x3, scalar, out=None, **kwargs):
        s = _scalar_for_dtype(scalar, x1)
        result = list(torch._foreach_addcdiv(x1, x2, x3, s))
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return result

    tolerance = _TOLERANCE


class TorchForeachAddcdivScalarSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, x3, scalar, **kwargs):
        s = _scalar_for_dtype(scalar, x1)
        return list(torch._foreach_addcdiv(x1, x2, x3, s))

    tolerance = _TOLERANCE
