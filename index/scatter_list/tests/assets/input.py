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

import numpy as np

__input__ = {"kernel": {"scatter_list": "scatter_list_input"}}


def scatter_list_input(var_list, indice, updates, mask=None, *, axis=-2, **kwargs):
    """
    Input function for scatter_list.
    All the parameters (names and order) follow scatter_list_def.cpp without outputs.
    var is a TensorList (list of ndarrays); indice/updates/mask are ndarrays.

    Resample `indice` so every scatter window [start, start + S2) stays inside the
    var scatter axis. 1-D indice carries only the start offset; 2-D indice carries
    [start, len] -> keep len, clamp start so start + len <= var_axis_size.

    Returns:
        Input tensors (var_list, indice, updates[, mask])
    """
    var0 = np.asarray(var_list[0])
    updates = np.asarray(updates)
    nax = axis if axis >= 0 else updates.ndim + axis
    var_nax = nax - 1  # updates has extra leading list dim
    var_axis = var0.shape[var_nax]
    S2 = updates.shape[nax]
    B = len(var_list)

    indice = np.asarray(indice)
    dt = indice.dtype
    if indice.ndim == 2:
        lens = np.minimum(indice[:, 1], S2)
        lens = np.clip(lens, 1, max(1, var_axis))
        highs = np.maximum(var_axis - lens, 0) + 1
        starts = np.array([np.random.randint(0, h) for h in highs], dtype=dt)
        new = np.stack([starts, lens.astype(dt)], axis=1)
    else:
        high = max(var_axis - S2, 0) + 1
        new = np.random.randint(0, high, size=(B,)).astype(dt)
        new = np.reshape(new, indice.shape)
    out = [var_list, new, updates]
    if mask is not None:
        out.append(mask)
    return out
