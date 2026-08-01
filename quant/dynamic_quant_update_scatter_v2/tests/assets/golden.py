#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch CPU golden and input plugin for DynamicQuantUpdateScatterV2."""

import numpy as np
import torch
from ml_dtypes import int4 as _int4  # noqa: F401 - registers int4 dtype for TTK

__golden__ = {
    "kernel": {
        "dynamic_quant_update_scatter_v2": "__golden_dynamic_quant_update_scatter_v2"
    }
}
__input__ = {
    "kernel": {
        "dynamic_quant_update_scatter_v2": "__input_dynamic_quant_update_scatter_v2"
    }
}

INT4_RANGE_INV = 1.0 / 15.0
INT4_QUANT_MAX = 7.0
QUANT_EPSILON = 1.0e-12


def _wrap_to_int4(q):
    return ((q + 8) & 15) - 8


def _pack_int4(values):
    vals = np.asarray(values).astype(np.int8).reshape(-1)
    if vals.size % 2:
        vals = np.pad(vals, (0, 1), mode="constant")
    pairs = vals.reshape(-1, 2).view(np.uint8) & np.uint8(0x0F)
    return (pairs[:, 0] | (pairs[:, 1] << np.uint8(4))).astype(np.uint8, copy=False)


def _unpack_int4(packed, size):
    raw = np.asarray(packed, dtype=np.uint8).reshape(-1)
    unpacked = np.empty(raw.size * 2, dtype=np.uint8)
    unpacked[0::2] = raw & np.uint8(0x0F)
    unpacked[1::2] = (raw >> np.uint8(4)) & np.uint8(0x0F)
    return unpacked[:size].view(_int4)


def __golden_dynamic_quant_update_scatter_v2(
    x, indices, var, var_scale, var_offset, **kwargs
):
    x_t = torch.as_tensor(np.asarray(x).astype(np.float32), dtype=torch.float32)
    hidden = x_t.shape[-1]
    rows = x_t.reshape(-1, hidden)
    batch = rows.shape[0]

    out_var = np.array(var, copy=True)
    out_scale = np.array(var_scale, dtype=np.float32, copy=True)
    out_offset = np.array(var_offset, dtype=np.float32, copy=True)
    seq_len = out_var.shape[1] if out_var.ndim >= 2 else 0
    var_bytes = _pack_int4(out_var)
    scale_flat = out_scale.reshape(-1)
    offset_flat = out_offset.reshape(-1)
    indices_t = torch.as_tensor(np.asarray(indices), dtype=torch.int64).reshape(-1)

    range_inv = torch.tensor(INT4_RANGE_INV, dtype=torch.float32)
    quant_max = torch.tensor(INT4_QUANT_MAX, dtype=torch.float32)
    quant_eps = torch.tensor(QUANT_EPSILON, dtype=torch.float32)
    for b in range(batch):
        row = rows[b]
        max_val = torch.max(row)
        min_val = torch.min(row)
        scale = torch.maximum((max_val - min_val) * range_inv, quant_eps)
        offset_quant = quant_max - max_val / scale
        quantized = torch.round(row / scale + offset_quant)
        quantized_i64 = _wrap_to_int4(quantized.to(torch.int64))

        dst = b * seq_len + int(indices_t[b].item())
        packed = _pack_int4(quantized_i64.numpy())
        byte_base = dst * hidden // 2
        byte_end = min(byte_base + packed.size, var_bytes.size)
        if 0 <= byte_base < byte_end:
            var_bytes[byte_base:byte_end] = packed[: byte_end - byte_base]
        scale_flat[dst] = np.float32(scale.item())
        offset_flat[dst] = np.float32(-offset_quant.item())

    out_var = _unpack_int4(var_bytes, out_var.size).reshape(out_var.shape)
    return [
        out_var,
        out_scale,
        out_offset,
    ]


def __input_dynamic_quant_update_scatter_v2(
    x, indices, var, var_scale, var_offset, **kwargs
):
    seq_len = var.shape[1] if var.ndim >= 2 else 0
    if seq_len > 0:
        clamped = (np.asarray(indices) % seq_len).astype(indices.dtype, copy=False)
    else:
        clamped = np.zeros_like(indices)
    return [x, clamped, var, var_scale, var_offset]
