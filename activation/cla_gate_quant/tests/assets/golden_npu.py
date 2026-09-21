#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""NPU small-op golden for cla_gate_quant.

This plugin runs the small-op reference path on the NPU itself:
    torch.sigmoid(gate).unsqueeze(-1) -> broadcast mul -> FMA(fused mul+add)
    -> reshape [T, N*D]
    -> torch_npu.npu_dynamic_mx_quant_with_dual_axis(...)

The merge uses the same single-rounding FP32 FMA as the fused kernel so the
reference stays bit-exact with its VF-fused `sl * Ol + sg * Og`.

TTK's Promote mode (mix_tolerance) upcasts fp16/bf16 inputs to fp32 before the
golden runs.  `_resolve_op_input_dtype` recovers the operator's declared dtype
from that promoted data, otherwise the golden would quantize a bf16-rounded
merge and diverge from the kernel on every fp16 case.

FP8/FP4 both use the NPU small-op reference.  For FP4, torch_npu returns the
data tensors already in packed uint8 layout (two 4-bit values per byte), so the
golden keeps that packed layout for TTK's uint8 FP4 cases.

单轴（dual_axis_flag=False，也是算子默认值）：不另走路，直接复用双轴小算子的
row_data / row_scale，并把 col_data / col_scale 置为空张量 —— 单轴小算子
npu_dynamic_mx_quant 存在精度问题，不能作为单轴融合算子的参考。

Use it only with the real-device backend (--backend npu). It cannot be used
with --backend npusim because NPUSim golden generation runs on CPU.
"""

import os

import numpy as np

# The NPU golden intentionally loads torch_npu.
import torch
import torch_npu  # noqa: F401


def _dst_torch_dtype(dst_type):
    if dst_type == 35:
        return getattr(torch, "float8_e5m2", None) or getattr(
            torch_npu, "float8_e5m2", None
        )
    if dst_type == 36:
        return getattr(torch, "float8_e4m3fn", None) or getattr(
            torch_npu, "float8_e4m3fn", None
        )
    if dst_type == 40:
        # torch_npu.npu_dynamic_mx_quant_with_dual_axis expects the torch_npu
        # FP4 enum (296 = float4_e2m1fn_x2), not GE dst_type 40. Passing the
        # dtype object makes op-plugin expand packed output to 2x width, and
        # passing 40 is interpreted as int4; use the torch_npu enum integer.
        return 296
    if dst_type == 41:
        return 297
    raise ValueError("unsupported dst_type for NPU small-op: %s" % dst_type)


def _data_numpy_dtype(dst_type):
    if dst_type == 35:
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if dst_type == 36:
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if dst_type == 40:
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if dst_type == 41:
        from en_dtypes import float4_e1m2

        return float4_e1m2
    raise ValueError("unsupported dst_type: %s" % dst_type)


def _e8m0_numpy_dtype():
    from en_dtypes import float8_e8m0

    return float8_e8m0


def _to_npu(array, device, torch_dtype=None):
    array = np.ascontiguousarray(array)
    if torch_dtype is None:
        torch_dtype = _resolve_op_input_dtype([array])
    if torch_dtype == torch.float16:
        return torch.from_numpy(array.astype(np.float16)).to(device)
    if torch_dtype == torch.bfloat16:
        # torch.from_numpy does not accept ml_dtypes.bfloat16. BF16 -> FP32 ->
        # BF16 is exact for every BF16 value, so this is lossless.
        return torch.from_numpy(array.astype(np.float32)).to(torch.bfloat16).to(device)
    return torch.from_numpy(array.astype(np.float32)).to(device)


def _resolve_op_input_dtype(arrays):
    """Recover the operator's declared input dtype from the arrays TTK passes.

    TTK's Promote mode (used by mix_tolerance) upcasts fp16/bf16 inputs to fp32
    before calling the golden, so ``array.dtype`` no longer tells fp16 from bf16.
    The fused kernel computes in the declared dtype, therefore the golden must
    cast back to it or every low-precision rounding decision is wrong.

    The source dtype is recoverable from the bit pattern of the promoted fp32
    values: an fp16 value keeps 10 mantissa bits (fp32 bits 13..22), so its low
    13 bits are zero and bits 13..15 may be set; a bf16 value keeps 7 mantissa
    bits, so its low 16 bits are zero.  If every value happens to be
    bf16-representable, casting to bf16 is exact anyway.
    """
    for array in arrays:
        if array is None:
            continue
        dtype = np.asarray(array).dtype
        if dtype == np.float16:
            return torch.float16
        if dtype == np.float64:
            return torch.float64
        if dtype == np.float32:
            bits = np.asarray(array, dtype=np.float32).view(np.uint32)
            if bits.size:
                if np.any((bits & 0x1FFF) != 0):
                    return torch.float32
                if np.any((bits & 0xE000) != 0):
                    return torch.float16
            return torch.bfloat16
        return torch.bfloat16
    return None


def _torch_uint8_bytes(tensor):
    return tensor.cpu().view(torch.uint8).numpy().tobytes()


def _fma_fp32(a, b, c):
    """Correctly-rounded FP32 fused multiply-add: round_to_nearest(a*b + c).

    The VF compiler fuses the trailing Mul+Add of the gate merge into a single
    instruction that keeps the second product at full precision before the add
    (``merged = fma(sl, Ol, sg*Og)``).  Emulating the FMA here is required: a
    plain ``a * b + c`` rounds the product first and disagrees with the kernel
    on inputs sitting near a rounding boundary.
    """
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    c64 = np.asarray(c, dtype=np.float64)

    # A product of two FP32 values is exactly representable in FP64, so the
    # only inexact step is the sum, which TwoSum splits into (s, err).
    product = a64 * b64
    s = product + c64
    bp = s - product
    err = (c64 - bp) + (product - (s - bp))

    f32 = s.astype(np.float32)
    f = f32.astype(np.float64)
    residual = (s - f) + err

    half_ulp = np.abs(np.spacing(f32)).astype(np.float64) * 0.5
    above = residual > 0.0
    below = residual < 0.0
    midpoint = np.where(above, f + half_ulp, f - half_ulp)
    to_midpoint = (s - midpoint) + err

    round_up = above & (to_midpoint > 0.0)
    round_down = below & (to_midpoint < 0.0)
    tie = (residual != 0.0) & (to_midpoint == 0.0)

    inf32 = np.float32(np.inf)
    next_up = np.nextafter(f32, np.where(above, inf32, np.float32(-np.inf)))
    next_down = np.nextafter(f32, np.float32(-np.inf))
    # Ties-to-even between f32 and its neighbour.
    tie_value = np.where((f32.view(np.int32) & 1) == 0, f32, next_up)

    result = np.where(round_up, next_up, f32)
    result = np.where(round_down, next_down, result)
    return np.where(tie, tie_value, result).astype(np.float32)


def _get_device_id():
    """Select the NPU device used by the golden.

    TTK_GOLDEN_DEVICE_ID is set by run_ttk_kernel_single_axis.sh from
    --device-whitelist. If it is present, use only that device.
    """
    explicit = os.environ.get("TTK_GOLDEN_DEVICE_ID", "").strip()
    if explicit.isdigit():
        dev_id = int(explicit)
        try:
            torch.npu.set_device(dev_id)
            _ = torch.zeros(1, device=f"npu:{dev_id}")
            return dev_id
        except Exception:
            pass

    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if visible:
        for item in visible.split(","):
            item = item.strip()
            if not item.isdigit():
                continue
            dev_id = int(item)
            try:
                torch.npu.set_device(dev_id)
                _ = torch.zeros(1, device=f"npu:{dev_id}")
                return dev_id
            except Exception:
                continue

    candidates = []
    try:
        current = torch.npu.current_device()
        if current is not None and int(current) >= 0:
            candidates.append(int(current))
    except Exception:
        pass

    try:
        device_count = torch.npu.device_count()
    except Exception:
        device_count = 0
    for dev_id in range(max(device_count, 1)):
        if dev_id not in candidates:
            candidates.append(dev_id)

    for dev_id in candidates:
        try:
            torch.npu.set_device(dev_id)
            _ = torch.zeros(1, device=f"npu:{dev_id}")
            return dev_id
        except Exception:
            continue

    return 0


def _unpack_packed_fp4(tensor, dst_type):
    """Unpack a torch_npu packed uint8 FP4 tensor to en_dtypes logical shape."""
    from ttk.utilities import unpack_4bits

    np_dtype = _data_numpy_dtype(dst_type)
    raw = np.frombuffer(_torch_uint8_bytes(tensor), dtype=np.uint8)
    unpacked = unpack_4bits(raw, np_dtype)
    logical_shape = list(tensor.shape)
    logical_shape[-1] *= 2
    return unpacked.reshape(logical_shape)


def cla_gate_quant_dual_npu_tensors(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    scale_alg=1,
    dst_type=36,
    round_mode="rint",
):
    """Run the NPU small-op reference and return raw torch tensors.

    Supports FP8 (dst_type=35/36) and FP4 (dst_type=40/41).  FP4 data tensors
    returned by torch_npu are packed uint8 with half the last dimension.
    """
    if dst_type not in (35, 36, 40, 41):
        raise ValueError(
            "NPU small-op golden path only supports dst_type=35/36/40/41, got %s"
            % dst_type
        )

    device_id = _get_device_id()
    device = f"npu:{device_id}"

    input_torch_dtype = _resolve_op_input_dtype(
        [global_out, local_out, global_gate_logits, local_gate_logits]
    )
    go = _to_npu(global_out, device, input_torch_dtype)
    lo = _to_npu(local_out, device, input_torch_dtype)
    gg = _to_npu(global_gate_logits, device, input_torch_dtype)
    lg = _to_npu(local_gate_logits, device, input_torch_dtype)
    if gg.dim() == go.dim():
        gg = gg.squeeze(-1)
        lg = lg.squeeze(-1)

    orig_dtype = go.dtype
    go = go.to(torch.float32)
    lo = lo.to(torch.float32)
    gg = gg.to(torch.float32)
    lg = lg.to(torch.float32)
    global_scale = torch.sigmoid(gg).unsqueeze(-1)
    local_scale = torch.sigmoid(lg).unsqueeze(-1)
    # Mirror the kernel's fused section: the first Mul is rounded to FP32 and
    # the second Mul+Add is a single-rounding FMA.  See _fma_fp32().
    global_term = (go.cpu().numpy() * global_scale.cpu().numpy()).astype(np.float32)
    merged_np = _fma_fp32(lo.cpu().numpy(), local_scale.cpu().numpy(), global_term)
    merged = torch.from_numpy(merged_np).to(device).to(orig_dtype)
    if merged.dim() == 3:
        # TND layout: [T, N, D] -> [T, N*D]
        merged = merged.reshape(merged.shape[0], merged.shape[1] * merged.shape[2])
    elif merged.dim() == 4:
        # Legacy SBND layout: [S, B, N, D] -> [S*B, N*D]
        merged = merged.reshape(
            merged.shape[0] * merged.shape[1], merged.shape[2] * merged.shape[3]
        )
    else:
        raise ValueError("cla_gate_quant expects 3D [T,N,D] or 4D [S,B,N,D]")

    row_data, row_scale, col_data, col_scale = (
        torch_npu.npu_dynamic_mx_quant_with_dual_axis(
            merged,
            round_mode=round_mode,
            dst_type=_dst_torch_dtype(dst_type),
            scale_alg=scale_alg,
        )
    )
    torch.npu.synchronize()
    return row_data, row_scale, col_data, col_scale


def cla_gate_quant_dual_npu(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    scale_alg=1,
    dst_type=36,
    round_mode="rint",
    dual_axis_flag=False,
):
    row_data, row_scale, col_data, col_scale = cla_gate_quant_dual_npu_tensors(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg=scale_alg,
        dst_type=dst_type,
        round_mode=round_mode,
    )

    if dst_type in (35, 36):
        row_data_np = np.frombuffer(
            _torch_uint8_bytes(row_data), dtype=_data_numpy_dtype(dst_type)
        ).reshape(row_data.shape)
        col_data_np = np.frombuffer(
            _torch_uint8_bytes(col_data), dtype=_data_numpy_dtype(dst_type)
        ).reshape(col_data.shape)
    else:
        # TTK CSV keeps the logical FP4 dtype/shape, so unpack the NPU small-op
        # packed uint8 outputs back to full float4 numpy arrays.
        row_data_np = _unpack_packed_fp4(row_data, dst_type)
        col_data_np = _unpack_packed_fp4(col_data, dst_type)

    row_scale_np = np.frombuffer(
        _torch_uint8_bytes(row_scale), dtype=_e8m0_numpy_dtype()
    ).reshape(row_scale.shape)
    col_scale_np = np.frombuffer(
        _torch_uint8_bytes(col_scale), dtype=_e8m0_numpy_dtype()
    ).reshape(col_scale.shape)
    if not dual_axis_flag:
        # Single-axis mode outputs empty col_data/col_scale.
        col_data_np = np.empty((0,), dtype=col_data_np.dtype)
        col_scale_np = np.empty((0,), dtype=col_scale_np.dtype)

    return row_data_np, row_scale_np, col_data_np, col_scale_np


def cla_gate_quant_npu_golden(
    global_out,
    local_out,
    global_gate_logits,
    local_gate_logits,
    *,
    dst_type=36,
    scale_alg=1,
    round_mode="rint",
    **kwargs,
):
    """Same-device NPU small-op golden reference for the CLA gate fused operator.

    Defaults must match the operator attributes (op_def / proto / torch schema):
    dst_type=36, scale_alg=1, round_mode="rint", dual_axis_flag=False（单轴）.
    """
    if dst_type not in (35, 36, 40, 41):
        raise ValueError("unsupported dst_type: %s" % dst_type)
    if dst_type in (35, 36) and round_mode != "rint":
        raise ValueError("FP8 only supports round_mode=rint")
    if dst_type in (40, 41) and scale_alg != 0:
        raise ValueError("FP4 only supports scale_alg=0")

    dual_axis_flag = bool(kwargs.get("dual_axis_flag", False))
    return cla_gate_quant_dual_npu(
        global_out,
        local_out,
        global_gate_logits,
        local_gate_logits,
        scale_alg=scale_alg,
        dst_type=dst_type,
        round_mode=round_mode,
        dual_axis_flag=dual_axis_flag,
    )


__golden__ = {"kernel": {"cla_gate_quant": "cla_gate_quant_npu_golden"}}
