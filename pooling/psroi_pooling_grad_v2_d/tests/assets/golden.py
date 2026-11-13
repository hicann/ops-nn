#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Binary32 CPU reference for ``psroi_pooling_grad_v2_d``.

The scalar algorithm matches the archived PSROIPoolingGradV2D reference:
round every arithmetic operation to binary32 and accumulate in fixed ROI order.
"""

import math
import struct

import numpy as np


__spec__ = {
    "psroi_pooling_grad_v2_d": "PSROIPoolingGradV2DTestSpec",
}


_BINARY32 = struct.Struct("<f")
_F32_ZERO = 0.0
_F32_ONE = 1.0
_F32_MIN_ROI_SIZE = _BINARY32.unpack(_BINARY32.pack(0.1))[0]


def _f32(value):
    """Round a Python numeric value to IEEE-754 binary32, ties-to-even."""
    value = float(value)
    try:
        return _BINARY32.unpack(_BINARY32.pack(value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def _f32_mul(a, b):
    """Execute one binary32 multiply and round immediately."""
    return _f32(_f32(a) * _f32(b))


def _f32_add(a, b):
    """Execute one binary32 add and round immediately."""
    return _f32(_f32(a) + _f32(b))


def _f32_sub(a, b):
    """Execute one binary32 subtract and round immediately."""
    return _f32(_f32(a) - _f32(b))


def _f32_div(a, b):
    """Execute one binary32 divide and round immediately."""
    numerator = _f32(a)
    denominator = _f32(b)
    if denominator == _F32_ZERO:
        if numerator == _F32_ZERO or math.isnan(numerator):
            return math.nan
        sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
        return math.copysign(math.inf, sign)
    return _f32(numerator / denominator)


def _rn_even_i32(value):
    """Round a finite binary32 value to an integer with ties-to-even."""
    return int(round(_f32(value)))


def _clamp_floor(value, upper):
    """Clamp a binary32 floor boundary before converting it to Python int."""
    value = _f32(value)
    if not value > _F32_ZERO:
        return 0
    if value >= upper:
        return upper
    return math.floor(value)


def _clamp_ceil(value, upper):
    """Clamp a binary32 ceil boundary before converting it to Python int."""
    value = _f32(value)
    if not value > _F32_ZERO:
        return 0
    if value >= upper:
        return upper
    return math.ceil(value)


def _validate_finite_rois(rois):
    for value in rois.flat:
        if not math.isfinite(float(value)):
            raise ValueError("rois must be finite")


def _validate_nonnegative_coordinates(rois, batch, rois_per_batch):
    for b in range(batch):
        for coordinate in range(1, 5):
            for r in range(rois_per_batch):
                if rois[b, coordinate, r] < _F32_ZERO:
                    raise ValueError("ROI coordinates must be >= 0")


def _validated_batch_ids(rois, batch, rois_per_batch):
    batch_ids = []
    for b in range(batch):
        row = []
        for r in range(rois_per_batch):
            value = _f32(rois[b, 0, r])
            if value != _f32(round(value)):
                raise ValueError("batch_id must be integer-valued")
            if value < _F32_ZERO or value >= batch:
                raise ValueError("batch_id out of range")
            row.append(int(value))
        batch_ids.append(row)
    return batch_ids


def golden_psroi_pooling_grad_v2_d(
    x,
    rois,
    spatial_scale,
    output_dim,
    group_size,
    input_size,
    **kwargs,
):
    """Return the deterministic PSROIPoolingGradV2D CPU reference tensor."""
    del kwargs
    x = np.asarray(x)
    rois = np.asarray(rois)
    if x.dtype != np.float32 or rois.dtype != np.float32:
        raise TypeError("x and rois must be float32")
    if x.ndim != 4 or rois.ndim != 3 or rois.shape[1] != 5:
        raise ValueError("invalid rank/rois shape")

    scale = _f32(spatial_scale)
    if not math.isfinite(scale) or scale <= _F32_ZERO:
        raise ValueError("spatial_scale must be finite and > 0")
    if output_dim <= 0 or not 1 <= group_size < 128:
        raise ValueError("invalid output_dim/group_size")
    if len(input_size) != 2 or input_size[0] <= 0 or input_size[1] <= 0:
        raise ValueError("input_size must be positive [H,W]")

    batch, _, rois_per_batch = rois.shape
    height, width = int(input_size[0]), int(input_size[1])
    if x.shape != (batch * rois_per_batch, output_dim, group_size, group_size):
        raise ValueError("x shape inconsistent with rois/attrs")
    _validate_finite_rois(rois)
    _validate_nonnegative_coordinates(rois, batch, rois_per_batch)
    batch_ids = _validated_batch_ids(rois, batch, rois_per_batch)

    group = int(group_size)
    output_shape = (batch, output_dim * group * group, height, width)
    output_bytes = batch * output_dim * group * group * height * width * 4
    y = np.ndarray(output_shape, dtype=np.float32, buffer=bytearray(output_bytes))

    # Fixed b/r order is part of the deterministic acceptance contract.
    for b in range(batch):
        for r in range(rois_per_batch):
            q = b * rois_per_batch + r
            roi_batch = batch_ids[b][r]
            start_w_i = _rn_even_i32(rois[b, 1, r])
            start_h_i = _rn_even_i32(rois[b, 2, r])
            end_w_i = _rn_even_i32(rois[b, 3, r]) + 1
            end_h_i = _rn_even_i32(rois[b, 4, r]) + 1

            start_w = _f32_mul(start_w_i, scale)
            start_h = _f32_mul(start_h_i, scale)
            end_w = _f32_mul(end_w_i, scale)
            end_h = _f32_mul(end_h_i, scale)
            roi_w = max(_f32_sub(end_w, start_w), _F32_MIN_ROI_SIZE)
            roi_h = max(_f32_sub(end_h, start_h), _F32_MIN_ROI_SIZE)
            bin_w = _f32_div(roi_w, group)
            bin_h = _f32_div(roi_h, group)

            for ph in range(group):
                # These remain separate binary32 operations: fusing mul+add
                # can change the floor/ceil support at exact boundaries.
                hs_f = _f32_add(_f32_mul(ph, bin_h), start_h)
                he_f = _f32_add(_f32_mul(ph + 1, bin_h), start_h)
                hs = _clamp_floor(hs_f, height)
                he = _clamp_ceil(he_f, height)
                for pw in range(group):
                    ws_f = _f32_add(_f32_mul(pw, bin_w), start_w)
                    we_f = _f32_add(_f32_mul(pw + 1, bin_w), start_w)
                    ws = _clamp_floor(ws_f, width)
                    we = _clamp_ceil(we_f, width)
                    area = (he - hs) * (we - ws)
                    if area <= 0:
                        continue
                    inv_area = _f32_div(_F32_ONE, area)
                    for output_channel in range(output_dim):
                        channel = output_channel * group * group + ph * group + pw
                        value = _f32_mul(x[q, output_channel, ph, pw], inv_area)
                        for h in range(hs, he):
                            for w in range(ws, we):
                                y[roi_batch, channel, h, w] = _f32_add(
                                    y[roi_batch, channel, h, w], value
                                )
    return y


def golden_ps_roi_pooling_grad_v2(
    x,
    rois,
    spatial_scale,
    output_dim,
    group_size,
    input_size,
    **kwargs,
):
    """Adapt the scalar reference to the existing single-output Golden API."""
    return [
        golden_psroi_pooling_grad_v2_d(
            x, rois, spatial_scale, output_dim, group_size, input_size, **kwargs
        )
    ]


class PSROIPoolingGradV2DTestSpec:
    """Kernel/GEIR reference with the design's elementwise close criterion."""

    golden = staticmethod(golden_ps_roi_pooling_grad_v2)

    # This operator specifies binary32 intermediate results, including sums.
    # Cross-check promotion to float64 would change the reference semantics.
    tolerance = {
        "float32": {
            "standard": "close",
            "rtol": 1e-4,
            "atol": 1e-8,
            "ptol": 0.0,
        },
    }
