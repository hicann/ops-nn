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
foreach_div_list 两方 Golden（无 GPU 版）

由用户 golden 适配而来，用于"NPU vs CPU golden"两方精度比对：
来源: /data/w50064389/SIMT-LongTails/ForeachDivList/deliverables/thirdparty_golden/golden.py

适配内容（相对原文件）：
1. tolerance: cross_check/L1 → mix_tolerance（生态混合容差标准；无需三方参考即可出 PASS/FAIL）
2. kernel golden 增加 bf16 (ml_dtypes 载体 / numpy 'V' dtype) 处理：
   view 还原 → fp32 计算 → 输出转回用例输出 dtype；修复原版 bf16 TypeError
3. 移除 third_party 声明：两方模式不需要，且原绑定方式不符 TTK 具名参数约定
"""

__spec__ = {
    "foreach_div_list": "ForeachDivListKernelSpec",
    # kernel CaseSet 的 op_name 使用 GE 算子名（ForeachDivList），与 opInterface 名（foreach_div_list）并存注册
    "ForeachDivList": "ForeachDivListKernelSpec",
    "aclnnForeachDivList": "AclnnForeachDivListSpec",
    "torch._foreach_div": "TorchForeachDivListSpec",
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
}


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


class ForeachDivListKernelSpec:
    """Kernel 路径 — list[np.ndarray] -> list[np.ndarray]。y[i] = x1[i] / x2[i]"""

    def golden(x1, x2, **kwargs):
        output_dtypes = _per_tensor_dtypes(kwargs.get("output_dtypes"))
        results = []
        for index, (a, b) in enumerate(zip(x1, x2)):
            ta = _to_torch_tensor(a)
            tb = _to_torch_tensor(b)
            out = torch._foreach_div([ta], [tb])[0]
            target = None
            if output_dtypes is not None and index < len(output_dtypes):
                target = _target_dtype(output_dtypes[index])
            if target is None:
                target = str(np.asarray(a).dtype)
            results.append(_cast_out(out, target))
        return results

    tolerance = _TOLERANCE


class AclnnForeachDivListSpec:
    """ACLNN 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, out=None, **kwargs):
        result = list(torch._foreach_div(x1, x2))
        if out is not None:
            for o, r in zip(out, result):
                o.copy_(r)
            return list(out)
        return result

    tolerance = _TOLERANCE


class TorchForeachDivListSpec:
    """E2E 路径 — list[torch.Tensor] -> list[torch.Tensor]。"""

    def golden(x1, x2, **kwargs):
        return list(torch._foreach_div(x1, x2))

    tolerance = _TOLERANCE
