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

"""TTK custom golden for scatter_min (kernel mode).

计算公式 (依据 aclnnScatterMin.md 「功能说明」节):
    varRef[indices[i], ...] = min(varRef[indices[i], ...], updates[i, ...])
  - indices 为扁平的索引条目序列; 对第 i 个索引条目, 取 var 的第 indices[i] 行
    (slice, 即 var[indices[i]]) 与 updates 的第 i 个 slice 逐元素取 min.
  - 越界索引被跳过 (scatter_reduce_common_simt.h:112: idxVal<0 或 >=var.shape[0] -> skip),
    与 kernel 语义一致.
  - 重复索引依次取 min (顺序无关, min 满足交换律/结合律).
  - 输出为原地更新后的 var (单输出).

输入顺序 (scatter_min_def.cpp): var, indices, updates
输出顺序 (scatter_min_def.cpp / infershape.cpp): var (inplace)
"""

import numpy as np


def __golden_scatter_min(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtype = var.dtype
    var_shape = var.shape

    # var.shape[0] = 索引上界; slice = var.shape[1:]
    var_first_dim = var_shape[0] if var_shape else 0
    slice_shape = tuple(var_shape[1:])
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # 空切片 (var.shape 含 0 维 -> slice_size==0): 每个 slice 0 宽, scatter 是 no-op,
    # var 原样返回 (kernel 同样在 sliceSize==0 时直接 return)。不加这道会让下面
    # upd_flat=(0,1) 在循环里 upd_flat[i] 越界 -> IndexError(golden 侧 GOLDEN_FAILURE)。
    if slice_size == 0:
        return [var.astype(out_dtype)]

    # 中间计算用 float32, 整型保持原类型精确比对
    is_float = np.issubdtype(out_dtype, np.floating)
    work_dtype = np.float32 if is_float else out_dtype

    result = (
        var.astype(work_dtype).reshape(var_first_dim, slice_size)
        if var_first_dim
        else var.astype(work_dtype).reshape(0, slice_size)
    )

    idx_flat = indices.reshape(-1).astype(np.int64)
    # updates 扁平化为 (indices_num, slice_size)
    upd_flat = (
        updates.astype(work_dtype).reshape(-1, slice_size)
        if slice_size
        else updates.astype(work_dtype).reshape(-1, 1)
    )

    n = idx_flat.shape[0]
    for i in range(n):
        m = int(idx_flat[i])
        if m < 0 or m >= var_first_dim:
            continue  # 越界跳过, 与 kernel 一致
        result[m, :] = np.minimum(result[m, :], upd_flat[i, :])

    out = result.reshape(var_shape).astype(out_dtype)
    return [out]


__golden__ = {"kernel": {"scatter_min": "__golden_scatter_min"}}
