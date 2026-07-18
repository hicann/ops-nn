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

"""TTK custom golden plugin for scatter_list (kernel mode).

Computation formula (docs/aclnnScatterList.md, 计算公式 section):
    var[i][.., indice[i] + k, ..] = updates[i][.., k, ..]   along `axis`
i.e. for each list element i, updates values are scattered into var along the
scatter axis starting at indice[i]. reduce='update' overwrites. An optional mask
(uint8, one entry per var-list element) gates whether each element is updated.

TTK passes each logical INPUT as one positional arg; a TensorList input arrives as
a Python list of ndarrays. Input order = scatter_list_def.cpp Input registration:
    var (TensorList of B tensors), indice, updates, mask(optional)
Output: var (TensorList, in-place).

updates.shape == [B] + var_tensor.shape (with scatter-axis size S2 <= var axis size),
so the scatter axis index on updates = axis, and on each var tensor = axis-relative
to the var rank (updates has one extra leading B dim).
"""

import numpy as np


def __golden_scatter_list(var_list, indice, updates, mask=None, **kwargs):
    axis = int(kwargs.get("axis", -2))

    var_list = [np.asarray(v) for v in var_list]
    indice = np.asarray(indice)
    updates = np.asarray(updates)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1)

    B = len(var_list)
    out = [v.astype(np.float32).copy() for v in var_list]

    # axis relative to updates dims; corresponding var axis is one less (updates has
    # the extra leading B dim).
    nax = axis if axis >= 0 else updates.ndim + axis
    var_nax = nax - 1
    S2 = updates.shape[nax]

    for i in range(B):
        if mask is not None and mask[i] == 0:
            continue
        # 1-D indice carries only the scatter start offset (kernel copies the full
        # updates scatter dim S2). 2-D indice carries [start, len]; the kernel copies
        # exactly `len` rows from updates (DataCopySmallPad: dim2UpdateLen), so the
        # golden must honor that length (capped at S2 to stay within updates).
        if indice.ndim == 2:
            start = int(indice[i][0])
            n_rows = min(int(indice[i][1]), S2)
        else:
            start = int(indice.reshape(-1)[i])
            n_rows = S2
        upd_i = updates[i].astype(np.float32)
        for k in range(n_rows):
            dst = [slice(None)] * out[i].ndim
            dst[var_nax] = start + k
            src = [slice(None)] * upd_i.ndim
            src[var_nax] = k
            out[i][tuple(dst)] = upd_i[tuple(src)]

    var_dtype = var_list[0].dtype
    return [o.astype(var_dtype) for o in out]


__golden__ = {"kernel": {"scatter_list": "__golden_scatter_list"}}
