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

"""TTK golden for foreach_log_inplace: 竞品 torch._foreach_log。
读法: 命名参数 x_list 直接迭代(同 add_list/div/a_cos)。bf16 升 fp32 算、输出 cast 回 bf16。有效域 (0,inf)。"""

import numpy as np
import torch

try:
    import ml_dtypes

    _BF16 = ml_dtypes.bfloat16
except ImportError:
    _BF16 = None


def _to_fp32(a):
    a = np.asarray(a)
    if a.dtype.kind == "V" and _BF16 is not None:
        a = a.view(_BF16)
    return a.astype(np.float32)


def __golden_foreach_log_inplace(x_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    tensors = [torch.from_numpy(_to_fp32(a)) for a in x_list]
    outs = torch._foreach_log(tensors)
    if output_dtypes and isinstance(output_dtypes[0], (list, tuple)):
        dt_flat = list(output_dtypes[0])
    else:
        dt_flat = list(output_dtypes or [])
    results = []
    for i, t in enumerate(outs):
        r = t.numpy()
        tgt = dt_flat[i] if i < len(dt_flat) else "float32"
        results.append(r.astype(_BF16) if tgt == "bfloat16" else r.astype(tgt))
    return results


__golden__ = {"kernel": {"foreach_log_inplace": "__golden_foreach_log_inplace"}}
