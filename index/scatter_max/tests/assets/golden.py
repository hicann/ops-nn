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

"""scatter_max TTK 自定义 golden plugin（kernel 模式）。

计算公式（aclnnScatterMax.md 计算公式节 / proto.h:36）:
  varRef[indices[i], ...] = max(varRef[indices[i], ...], updates[i, ...])
  - 多个 updates 作用到 var 同一切片时依次取最大值（顺序无关）。
  - shape 约束: updates.shape = indices.shape + var.shape[1:]。
  - indices 越界值 (idx < 0 或 idx >= var.shape[0]) 被算子跳过（kernel simt.h:112）。
  - var 原地更新，输出即更新后的 var。

输入顺序: (var, indices, updates)  输出: [var]
"""

import numpy as np


def __golden_scatter_max(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out = var.copy()
    var_first_dim = out.shape[0] if out.ndim >= 1 else 0

    idx_flat = indices.reshape(-1)
    n_idx = idx_flat.shape[0]

    if n_idx == 0 or var_first_dim == 0 or updates.size == 0:
        return [out.astype(var.dtype)]

    # updates 展平为 (n_idx, *slice_shape)：slice_shape = var.shape[1:]
    slice_shape = out.shape[1:]
    upd = updates.reshape((n_idx,) + tuple(slice_shape))

    # 浮点用 float32 中间计算（fp16 numpy 不自动提升）；整型保持原类型精确比对。
    # 注意：max 不累加、永不溢出，整型绝不能转 float32——大 int32(>2^24) 经 float32
    # round-trip 会丢精度，导致与 kernel 的精确 int32 max 不符（曾误报 inf/大值用例失败）。
    is_float = np.issubdtype(var.dtype, np.floating)
    work_dtype = np.float32 if is_float else var.dtype
    work = out.astype(work_dtype)
    upd_w = upd.astype(work_dtype)

    for i in range(n_idx):
        idx = int(idx_flat[i])
        if idx < 0 or idx >= var_first_dim:
            continue  # 越界跳过，与 kernel 一致
        work[idx] = np.maximum(work[idx], upd_w[i])

    return [work.astype(var.dtype)]


__golden__ = {"kernel": {"scatter_max": "__golden_scatter_max"}}
