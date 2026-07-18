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

"""TTK custom golden plugin for scatter_mul (kernel mode).

Compute formula (docs/aclnnScatterMul.md, 计算公式节):
    varRef[indices[i], ...] = varRef[indices[i], ...] * updates[i, ...]
若多个 updates 作用到同一切片，则在该切片上连乘。
索引越界（idx < 0 或 idx >= var.shape[0]）按 kernel 语义跳过。
"""

import numpy as np

_NP_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
    "uint8": np.uint8,
}


def __golden_scatter_mul(*input_arrays, **kwargs):
    # input order matches CSV input_shapes: var, indices, updates
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtype = var.dtype
    # accumulate in a wider type to match the kernel's internal fp32/int32 precision
    if np.issubdtype(out_dtype, np.floating):
        acc_dtype = np.float32
    else:
        acc_dtype = np.int64

    result = var.astype(acc_dtype).copy()
    upd = updates.astype(acc_dtype)

    var_first = result.shape[0] if result.ndim >= 1 else 1
    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates leading dims correspond to indices entries; trailing = var.shape[1:]
    n_idx = idx_flat.shape[0]
    slice_shape = result.shape[1:]
    upd_slices = (
        upd.reshape((n_idx,) + tuple(slice_shape))
        if n_idx > 0
        else upd.reshape((0,) + tuple(slice_shape))
    )

    for i in range(n_idx):
        idx = int(idx_flat[i])
        if idx < 0 or idx >= var_first:
            continue  # out-of-bound index skipped (scatter_reduce_common_simt.h:112)
        result[idx] = result[idx] * upd_slices[i]

    return [result.astype(out_dtype)]


__golden__ = {"kernel": {"scatter_mul": "__golden_scatter_mul"}}
