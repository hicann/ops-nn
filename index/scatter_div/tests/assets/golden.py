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

"""TTK custom golden for ScatterDiv (kernel mode).

计算公式 (docs/aclnnScatterDiv.md):
    varRef[indices[i], ...] = varRef[indices[i], ...] / updates[i, ...]
多个 updates 命中同一切片时连除（顺序无关，与 TF scatter_div 一致）。
越界索引 (idx < 0 或 idx >= var.shape[0]) 跳过 (scatter_reduce_common_simt.h:112)。
整型 dtype (int32/int8/uint8) 走 C++ 整数除法（向零截断）。
in-place: 输出 var 与输入 var 同缓冲（output_inplace_indexes=(0,)）。
"""

import numpy as np

_INT_DTYPES = {"int32", "int8", "uint8"}


def __golden_scatter_div(*input_arrays, **kwargs):
    var, indices, updates = input_arrays[0], input_arrays[1], input_arrays[2]

    out_dtypes = kwargs.get("output_dtypes", None)
    if out_dtypes:
        out_dt_str = out_dtypes[0]
    else:
        out_dt_str = str(var.dtype)
    is_int = out_dt_str in _INT_DTYPES

    var_first = var.shape[0]
    slice_shape = var.shape[1:]
    slice_size = int(np.prod(slice_shape)) if slice_shape else 1

    # work in float32 for fp, int64 for int (avoid overflow during division)
    if is_int:
        work = var.astype(np.int64).reshape(var_first, slice_size)
        upd = updates.astype(np.int64).reshape(-1, slice_size)
    else:
        work = var.astype(np.float32).reshape(var_first, slice_size)
        upd = updates.astype(np.float32).reshape(-1, slice_size)

    idx_flat = indices.reshape(-1).astype(np.int64)
    n = idx_flat.shape[0]

    for m in range(n):
        idv = int(idx_flat[m])
        if idv < 0 or idv >= var_first:
            continue  # out-of-bound skip
        if is_int:
            # C++ integer division (truncation toward zero)
            denom = upd[m]
            res = np.empty_like(work[idv])
            for j in range(slice_size):
                d = int(denom[j])
                if d == 0:
                    # match C++ UB conservatively: leave unchanged is not defined;
                    # use trunc-toward-zero with denom guarded to avoid python ZeroDivision.
                    res[j] = work[idv][j]
                else:
                    q = work[idv][j] / d
                    res[j] = int(q) if q >= 0 else -int(-q)
            work[idv] = res
        else:
            work[idv] = work[idv] / upd[m]

    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }.get(out_dt_str, var.dtype)

    out = work.reshape(var.shape).astype(np_dtype)
    return [out]


__golden__ = {"kernel": {"scatter_div": "__golden_scatter_div"}}
