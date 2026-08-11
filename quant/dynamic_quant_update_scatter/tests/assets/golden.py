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

"""Torch CPU golden plugin for DynamicQuantUpdateScatter."""

import numpy as np
import torch

__spec__ = {
    "dynamic_quant_update_scatter": "DynamicQuantUpdateScatterKernelSpec",
}

__golden__ = {
    "kernel": {"dynamic_quant_update_scatter": "dynamic_quant_update_scatter_golden"}
}
__input__ = {
    "kernel": {"dynamic_quant_update_scatter": "dynamic_quant_update_scatter_input"}
}

QUANT_MAX = 127.0

_KERNEL_TOLERANCE = {
    "int8": {"standard": "quant"},
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _norm_axis(axis, rank):
    normalized = axis + rank if axis < 0 else axis
    if normalized <= 0 or normalized >= rank - 1:
        raise ValueError(f"axis must be inner dim, got {normalized}")
    return normalized


def dynamic_quant_update_scatter_input(
    var,
    var_scale,
    indices,
    updates,
    smooth_scales=None,
    reduce="update",
    axis=0,
    **kwargs,
):
    axis_n = _norm_axis(axis, var.ndim)
    update_axis = updates.shape[axis_n]
    max_start = var.shape[axis_n] - update_axis
    if max_start < 0:
        raise ValueError("updates exceed var")
    update_batch = updates.shape[0]
    generated = np.zeros_like(indices)
    if indices.ndim == 2:
        generated[:, 0] = np.arange(update_batch, dtype=np.int64) % var.shape[0]
        generated[:, 1] = np.arange(update_batch, dtype=np.int64) % (max_start + 1)
    else:
        generated[...] = np.arange(update_batch, dtype=np.int64) % (max_start + 1)
    return [var, var_scale, generated.astype(indices.dtype), updates, smooth_scales]


def _to_torch(x, dtype=torch.float32):
    if x is None:
        return None
    arr = np.asarray(x)
    try:
        return torch.as_tensor(arr.copy(), dtype=dtype)
    except TypeError:
        return torch.as_tensor(arr.astype(np.float32).copy(), dtype=dtype)


def _to_numpy_for_golden(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def dynamic_quant_update_scatter_golden(
    var,
    var_scale,
    indices,
    updates,
    smooth_scales=None,
    reduce="update",
    axis=0,
    **kwargs,
):
    var_t = _to_torch(var).clone()
    scale_t = _to_torch(var_scale).clone()
    updates_t = _to_torch(updates)
    smooth_t = _to_torch(smooth_scales)
    index_arr = np.asarray(indices)

    axis_n = _norm_axis(axis, var_t.dim())
    var_batch = var_t.shape[0]
    update_batch = updates_t.shape[0]
    head = int(np.prod(tuple(var_t.shape[1:axis_n]), dtype=np.int64))
    axis_size = var_t.shape[axis_n]
    update_axis = updates_t.shape[axis_n]
    last_dim = var_t.shape[-1]
    middle = int(np.prod(tuple(updates_t.shape[axis_n + 1 :]), dtype=np.int64))
    quant_groups = middle // last_dim

    var_view = var_t.reshape(var_batch, head, axis_size, quant_groups, last_dim)
    scale_view = scale_t.reshape(var_batch, head, axis_size, quant_groups, 1)
    updates_view = updates_t.reshape(
        update_batch, head, update_axis, quant_groups, last_dim
    ).float()
    smooth_view = smooth_t.reshape(-1).float() if smooth_t is not None else None
    qmax = torch.tensor(QUANT_MAX, dtype=torch.float32)

    for b in range(update_batch):
        out_b = int(index_arr[b, 0]) if index_arr.ndim == 2 else b
        axis_start = int(index_arr[b, 1]) if index_arr.ndim == 2 else int(index_arr[b])
        for h in range(head):
            for u in range(update_axis):
                out_axis = axis_start + u
                for q in range(quant_groups):
                    row = updates_view[b, h, u, q]
                    if smooth_view is not None:
                        row = row * smooth_view
                    amax = torch.max(torch.abs(row))
                    if amax == 0:
                        multiplier = torch.zeros_like(amax)
                        output_scale = torch.zeros_like(amax)
                    else:
                        multiplier = qmax / amax
                        output_scale = 1.0 / multiplier
                    quantized = torch.round(row * multiplier)
                    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)
                    var_view[out_b, h, out_axis, q] = quantized.to(torch.float32)
                    scale_view[out_b, h, out_axis, q, 0] = output_scale

    return [
        var_t.numpy().astype(np.asarray(var).dtype, copy=False),
        scale_t.numpy().astype(np.float32, copy=False),
    ]


class _DynamicQuantUpdateScatterCompose:
    """Third-party reference executed on the remote GPU server."""

    def __init__(self, reduce="update", axis=0, **kwargs):
        self.reduce = reduce
        self.axis = axis

    def __call__(self, var, var_scale, indices, updates, smooth_scales=None, **kwargs):
        reduce = kwargs.get("reduce", self.reduce)
        axis = kwargs.get("axis", self.axis)
        outputs = dynamic_quant_update_scatter_golden(
            _to_numpy_for_golden(var),
            _to_numpy_for_golden(var_scale),
            _to_numpy_for_golden(indices),
            _to_numpy_for_golden(updates),
            _to_numpy_for_golden(smooth_scales),
            reduce=reduce,
            axis=axis,
            **kwargs,
        )
        device = var.device if torch.is_tensor(var) else "cpu"
        return [torch.as_tensor(out, device=device) for out in outputs]


class DynamicQuantUpdateScatterKernelSpec:
    golden = dynamic_quant_update_scatter_golden
    input = dynamic_quant_update_scatter_input
    third_party = {"torch": _DynamicQuantUpdateScatterCompose}
    tolerance = _KERNEL_TOLERANCE


# 【不存在】aclnn 通路: CMakeLists.txt 使用 ACLNNTYPE aclnn_exclude.
# 【不存在】e2e 通路: 未发现 torch_npu eager/aten 绑定到 DynamicQuantUpdateScatter.
