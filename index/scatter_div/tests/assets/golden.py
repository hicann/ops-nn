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

实现说明: 计算由 numpy/python 公式改为 torch 竞品算子拼接（index_select + torch.div +
index_copy_，整型走 rounding_mode="trunc" 即 C++ 向零截断）——纯 numpy 公式实现与被测
kernel 易犯同类错误，会掩盖 kernel 精度短板。numpy 仅保留 I/O 与 dtype 转换。
除法不满足交换律（a/b1/b2 != a/(b1*b2)，逐位不同），故仍按索引顺序逐条下发，
不能折叠成一次 index_reduce；语义（连除顺序、越界跳过、工作 dtype、返回结构）未变。
"""

import numpy as np
import torch

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
        work = torch.from_numpy(var.astype(np.int64).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(np.int64).reshape(-1, slice_size))
    else:
        work = torch.from_numpy(var.astype(np.float32).reshape(var_first, slice_size))
        upd = torch.from_numpy(updates.astype(np.float32).reshape(-1, slice_size))

    idx_flat = indices.reshape(-1).astype(np.int64)
    n = idx_flat.shape[0]

    for m in range(n):
        idv = int(idx_flat[m])
        if idv < 0 or idv >= var_first:
            continue  # out-of-bound skip
        row = work.index_select(0, torch.tensor([idv]))
        if is_int:
            # C++ integer division (truncation toward zero)
            denom = upd[m : m + 1]
            nonzero = denom != 0
            # match C++ UB conservatively: leave unchanged is not defined;
            # use trunc-toward-zero with denom guarded to avoid a division by zero.
            safe = torch.where(nonzero, denom, torch.ones_like(denom))
            res = torch.where(nonzero, torch.div(row, safe, rounding_mode="trunc"), row)
        else:
            res = torch.div(row, upd[m : m + 1])
        work.index_copy_(0, torch.tensor([idv]), res)

    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "bfloat16": np.float32,
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
    }.get(out_dt_str, var.dtype)

    out = work.numpy().reshape(var.shape).astype(np_dtype)
    return [out]


__golden__ = {"kernel": {"scatter_div": "__golden_scatter_div"}}
