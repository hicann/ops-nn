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

"""TTK golden for foreach_mul_list_inplace: 竞品 torch._foreach_mul (x1_i *= x2_i)。
读法: 命名参数 x1_list/x2_list 直接迭代(原 *input_arrays+n_total//2 会把整个list当1个张量, 错)。
bf16 升 fp32 算、输出 cast 回 bf16; int 保持原 dtype。"""

import numpy as np
import torch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None


def _to_compute(a):
    """bf16/void 升 fp32(torch 不收 bf16); 其它(int/fp16/fp32)保持。"""
    a = np.asarray(a)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16).astype(np.float32)
    return a


def __golden_foreach_mul_list_inplace(x1_list, x2_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    if output_dtypes and isinstance(output_dtypes[0], (list, tuple)):
        dt_flat = list(output_dtypes[0])
    else:
        dt_flat = list(output_dtypes or [])
    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        a = np.asarray(a)
        b = np.asarray(b)
        tgt = dt_flat[i] if i < len(dt_flat) else str(a.dtype)
        if np.issubdtype(a.dtype, np.integer):
            res = (a.astype(np.int64) * b.astype(np.int64)).astype(tgt)
        else:
            ta = torch.from_numpy(_to_compute(a))
            tb = torch.from_numpy(_to_compute(b))
            res = (ta * tb).numpy().astype(_BF16 if tgt == "bfloat16" else tgt)
        results.append(res)
    return results


__golden__ = {
    "kernel": {"foreach_mul_list_inplace": "__golden_foreach_mul_list_inplace"}
}
