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

实现说明: 计算由 numpy 逐条连乘改为 torch 竞品算子 Tensor.index_reduce_(reduce="prod",
include_self=True)——纯 numpy 公式实现与被测 kernel 易犯同类错误，会掩盖 kernel 精度短板。
numpy 仅保留 I/O 与 dtype 转换。语义（重复索引连乘、越界跳过、累加 dtype、返回结构）未变。
"""

import numpy as np
import torch

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

    # out-of-bound indices dropped first (scatter_reduce_common_simt.h:112), the rest
    # go through the torch reference op; include_self=True == var * all matching updates.
    result_t = torch.from_numpy(result)
    idx_t = torch.from_numpy(idx_flat)
    valid = (idx_t >= 0) & (idx_t < var_first)
    if int(valid.sum()) > 0:
        upd_t = torch.from_numpy(upd_slices)
        result_t.index_reduce_(0, idx_t[valid], upd_t[valid], "prod", include_self=True)

    return [result_t.numpy().astype(out_dtype)]


__golden__ = {"kernel": {"scatter_mul": "__golden_scatter_mul"}}
