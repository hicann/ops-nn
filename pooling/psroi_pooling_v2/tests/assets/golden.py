#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Golden and independent Torch references for ``psroi_pooling_v2``.

The NumPy-facing Kernel Golden delegates to a vectorized Torch implementation.
The remote provider uses an independent device-side Torch reference that
matches the NPU float32 reduction semantics.
"""

import math
import os
from collections.abc import Callable

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch


__spec__ = {
    "psroi_pooling_v2": "PsroiPoolingV2TestSpec",
    "PSROIPoolingV2": "PsroiPoolingV2TestSpec",
}


_OUTPUT_CHUNK_SIZE = 65_536
_SAMPLE_SLOT_BUDGET = 262_144
_SAMPLE_BLOCK_LIMIT = 4_096


def _kernel_bound(
    value: torch.Tensor,
    upper: int,
    round_op: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply the Kernel's float32 clamp-before-conversion boundary rule."""
    upper_float = torch.as_tensor(upper, dtype=torch.float32, device=value.device)
    positive = value > 0.0
    at_or_above_upper = value >= upper_float
    safe_value = torch.where(positive & ~at_or_above_upper, value, 0.0)
    converted = round_op(safe_value).to(torch.long).clamp_max(upper)
    return torch.where(
        positive,
        torch.where(at_or_above_upper, torch.full_like(converted, upper), converted),
        torch.zeros_like(converted),
    )


def _reduce_selected_regions(
    flat_x: torch.Tensor,
    selected_area: torch.Tensor,
    selected_width: torch.Tensor,
    selected_base: torch.Tensor,
    *,
    feature_width: int,
    sample_limit: int,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce selected regions while preserving the original summation order."""
    count = selected_area.numel()
    sample_block = min(sample_limit, max(1, _SAMPLE_SLOT_BUDGET // count))
    sums = torch.zeros(count, dtype=acc_dtype, device=flat_x.device)
    compensation = torch.zeros_like(sums)
    has_nan = torch.zeros(count, dtype=torch.bool, device=flat_x.device)
    has_positive_inf = torch.zeros_like(has_nan)
    has_negative_inf = torch.zeros_like(has_nan)
    for sample_begin in range(0, sample_limit, sample_block):
        sample_end = min(sample_begin + sample_block, sample_limit)
        offsets = torch.arange(
            sample_begin, sample_end, dtype=torch.long, device=flat_x.device
        )
        valid = offsets.unsqueeze(0) < selected_area.unsqueeze(1)
        rows = torch.div(
            offsets.unsqueeze(0), selected_width.unsqueeze(1), rounding_mode="floor"
        )
        cols = offsets.unsqueeze(0) - rows * selected_width.unsqueeze(1)
        sample_indices = selected_base.unsqueeze(1) + rows * feature_width + cols
        sample_indices = torch.where(
            valid, sample_indices, torch.zeros_like(sample_indices)
        )
        values = flat_x.index_select(0, sample_indices.reshape(-1)).reshape(valid.shape)
        has_nan |= torch.any(valid & torch.isnan(values), dim=1)
        has_positive_inf |= torch.any(valid & torch.isposinf(values), dim=1)
        has_negative_inf |= torch.any(valid & torch.isneginf(values), dim=1)
        terms = torch.where(
            valid & torch.isfinite(values),
            values.to(acc_dtype),
            torch.zeros((), dtype=acc_dtype, device=flat_x.device),
        )
        for term_index in range(terms.shape[1]):
            term = terms[:, term_index]
            updated = sums + term
            compensation += torch.where(
                torch.abs(sums) >= torch.abs(term),
                (sums - updated) + term,
                (term - updated) + sums,
            )
            sums = updated

    means = (sums + compensation) / selected_area.to(acc_dtype)
    conflicting_inf = has_positive_inf & has_negative_inf
    means = torch.where(
        has_nan | conflicting_inf, torch.full_like(means, float("nan")), means
    )
    means = torch.where(
        has_positive_inf & ~has_negative_inf & ~has_nan,
        torch.full_like(means, float("inf")),
        means,
    )
    return torch.where(
        has_negative_inf & ~has_positive_inf & ~has_nan,
        torch.full_like(means, -float("inf")),
        means,
    )


def _write_area_bucket_means(
    output: torch.Tensor,
    output_ids: torch.Tensor,
    flat_x: torch.Tensor,
    area: torch.Tensor,
    region_width: torch.Tensor,
    base: torch.Tensor,
    *,
    bucket_lower: int,
    bucket_upper: int,
    feature_width: int,
    acc_dtype: torch.dtype,
) -> None:
    """Reduce one area bucket in bounded position chunks and write its means."""
    positions = torch.nonzero(
        (area > bucket_lower) & (area <= bucket_upper), as_tuple=False
    ).flatten()
    if positions.numel() == 0:
        return
    max_positions = max(
        1, _SAMPLE_SLOT_BUDGET // min(bucket_upper, _SAMPLE_BLOCK_LIMIT)
    )
    for position_begin in range(0, positions.numel(), max_positions):
        selected = positions[position_begin : position_begin + max_positions]
        means = _reduce_selected_regions(
            flat_x,
            area.index_select(0, selected),
            region_width.index_select(0, selected),
            base.index_select(0, selected),
            feature_width=feature_width,
            sample_limit=bucket_upper,
            acc_dtype=acc_dtype,
        )
        output.index_copy_(
            0, output_ids.index_select(0, selected), means.to(output.dtype)
        )


def _target_region_means(
    x: torch.Tensor,
    batch_ids: torch.Tensor,
    hstart: torch.Tensor,
    hend: torch.Tensor,
    wstart: torch.Tensor,
    wend: torch.Tensor,
    output_dim: int,
    group_size: int,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    """Reduce only selected bin elements with bounded temporary storage."""
    n, channels, height, width = x.shape
    roi_count = batch_ids.shape[1]
    group_area = group_size * group_size
    total_outputs = n * roi_count * output_dim * group_area
    output = x.new_zeros((total_outputs,))
    if total_outputs == 0:
        return output

    flat_x = x.to(acc_dtype).reshape(-1)
    flat_batch_ids = batch_ids.reshape(-1)
    flat_hstart = hstart.reshape(n * roi_count, group_size)
    flat_hend = hend.reshape(n * roi_count, group_size)
    flat_wstart = wstart.reshape(n * roi_count, group_size)
    flat_wend = wend.reshape(n * roi_count, group_size)
    for output_begin in range(0, total_outputs, _OUTPUT_CHUNK_SIZE):
        output_end = min(output_begin + _OUTPUT_CHUNK_SIZE, total_outputs)
        output_ids = torch.arange(
            output_begin, output_end, dtype=torch.long, device=x.device
        )
        roi_slots = torch.div(output_ids, channels, rounding_mode="floor")
        channel_ids = output_ids - roi_slots * channels
        bin_ids = channel_ids % group_area
        ph = torch.div(bin_ids, group_size, rounding_mode="floor")
        pw = bin_ids - ph * group_size
        source_batch = flat_batch_ids.index_select(0, roi_slots)
        hs = flat_hstart[roi_slots, ph]
        he = flat_hend[roi_slots, ph]
        ws = flat_wstart[roi_slots, pw]
        we = flat_wend[roi_slots, pw]
        region_width = we - ws
        area = (he - hs) * region_width
        base = ((source_batch * channels + channel_ids) * height + hs) * width + ws
        max_area = int(area.max().item())
        bucket_lower = 0
        bucket_upper = 1
        while bucket_lower < max_area:
            _write_area_bucket_means(
                output,
                output_ids,
                flat_x,
                area,
                region_width,
                base,
                bucket_lower=bucket_lower,
                bucket_upper=bucket_upper,
                feature_width=width,
                acc_dtype=acc_dtype,
            )
            bucket_lower = bucket_upper
            bucket_upper *= 2
    return output


def _psroi_pooling_v2_torch_core(
    x: torch.Tensor,
    rois: torch.Tensor,
    spatial_scale: float,
    output_dim: int,
    group_size: int,
) -> torch.Tensor:
    """Compute the NumPy-facing Golden with device-side Torch operations.

    ``x`` and ``rois`` are shape-validated Torch tensors on the same device.
    Invalid RoI values use the Kernel's per-RoI all-zero fallback. The remote
    provider uses a separate independent Torch reference.
    """
    n, _, h, w = x.shape
    r_count = rois.shape[2]
    if n == 0 or r_count == 0:
        return x.new_zeros(
            (
                n * r_count,
                output_dim,
                group_size,
                group_size,
            )
        )

    acc_dtype = torch.float32 if x.dtype == torch.float16 else torch.float64
    normalized_rois = rois.to(torch.float32)
    rounded_batch_ids = torch.round(normalized_rois[:, 0, :])
    valid_rois = (
        torch.all(torch.isfinite(normalized_rois), dim=1)
        & torch.all(normalized_rois[:, 1:5, :] >= 0, dim=1)
        & (rounded_batch_ids >= 0)
        & (rounded_batch_ids < n)
    )
    safe_rois = torch.where(
        valid_rois.unsqueeze(1), normalized_rois, torch.zeros_like(normalized_rois)
    )
    rounded = torch.round(safe_rois)
    batch_ids = rounded[:, 0, :].to(torch.long)
    scale = torch.as_tensor(spatial_scale, dtype=torch.float32, device=rois.device)
    roi_start_w = rounded[:, 1, :] * scale
    roi_start_h = rounded[:, 2, :] * scale
    roi_end_w = (rounded[:, 3, :] + 1.0) * scale
    roi_end_h = (rounded[:, 4, :] + 1.0) * scale
    minimum_extent = torch.as_tensor(0.1, dtype=torch.float32, device=x.device)
    width_delta = roi_end_w - roi_start_w
    height_delta = roi_end_h - roi_start_h
    roi_w = torch.where(width_delta > minimum_extent, width_delta, minimum_extent)
    roi_h = torch.where(height_delta > minimum_extent, height_delta, minimum_extent)
    bin_w = roi_w / group_size
    bin_h = roi_h / group_size

    grid = torch.arange(
        group_size,
        dtype=torch.float32,
        device=x.device,
    )
    hstart = _kernel_bound(
        roi_start_h.unsqueeze(-1) + grid * bin_h.unsqueeze(-1), h, torch.floor
    )
    hend = _kernel_bound(
        roi_start_h.unsqueeze(-1) + (grid + 1.0) * bin_h.unsqueeze(-1), h, torch.ceil
    )
    wstart = _kernel_bound(
        roi_start_w.unsqueeze(-1) + grid * bin_w.unsqueeze(-1), w, torch.floor
    )
    wend = _kernel_bound(
        roi_start_w.unsqueeze(-1) + (grid + 1.0) * bin_w.unsqueeze(-1), w, torch.ceil
    )

    output = _target_region_means(
        x,
        batch_ids,
        hstart,
        hend,
        wstart,
        wend,
        output_dim,
        group_size,
        acc_dtype,
    )
    output = output.reshape(n, r_count, output_dim, group_size, group_size)
    output = torch.where(
        valid_rois.reshape(n, r_count, 1, 1, 1),
        output,
        torch.zeros((), dtype=x.dtype, device=x.device),
    )
    return output.reshape(n * r_count, output_dim, group_size, group_size)


def psroi_pooling_v2_golden(
    x,
    rois,
    *,
    spatial_scale,
    output_dim,
    group_size,
    **kwargs,
):
    """Golden function for psroi_pooling_v2.

    Parameters follow the PSROIPoolingV2 REG_OP definition in SE section 6.2,
    without the output tensor. All Tensor inputs are ``numpy.ndarray``.

    Args:
        x: Position-sensitive feature map with shape ``[N, C, H, W]``.
        rois: RoI tensor with shape ``[N, 5, R]``.
        spatial_scale: Positive finite coordinate scale.
        output_dim: Positive output channel count.
        group_size: Position-sensitive group size in ``[1, 127]``.
        **kwargs: TTK metadata. ``golden_mode="Promote"`` also permits float64.

    Returns:
        A one-element list containing the NumPy output array with shape
        ``[N * R, output_dim, group_size, group_size]``.
    """
    golden_mode = kwargs.get("golden_mode")
    del kwargs
    x = np.asarray(x)
    rois = np.asarray(rois)

    supported = (np.dtype(np.float16), np.dtype(np.float32))
    # TTK promotes float32 to float64 for the high-precision Golden only.
    if golden_mode == "Promote":
        supported += (np.dtype(np.float64),)
    if x.dtype not in supported or rois.dtype != x.dtype:
        raise TypeError(
            "x/rois must have the same float16 or float32 dtype "
            "(float64 is allowed only in Promote mode)"
        )
    if x.ndim != 4:
        raise ValueError("x must be a rank-4 ND tensor [N,C,H,W]")
    if rois.ndim != 3 or rois.shape[1] != 5:
        raise ValueError("rois must have shape [N,5,R]")

    n, c, h, w = x.shape
    if rois.shape[0] != n:
        raise ValueError("rois.shape[0] must equal x.shape[0]")
    if h <= 0 or w <= 0:
        raise ValueError("H and W must be greater than zero")
    if not math.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive and finite")
    if int(output_dim) != output_dim or output_dim <= 0:
        raise ValueError("output_dim must be a positive integer")
    if int(group_size) != group_size or not 1 <= group_size < 128:
        raise ValueError("group_size must be an integer in [1, 127]")

    output_dim = int(output_dim)
    group_size = int(group_size)
    expected_c = output_dim * group_size * group_size
    if c != expected_c:
        raise ValueError("C must equal output_dim*group_size^2")
    torch_x = torch.from_numpy(np.ascontiguousarray(x))
    torch_rois = torch.from_numpy(np.ascontiguousarray(rois))
    output = _psroi_pooling_v2_torch_core(
        torch_x,
        torch_rois,
        float(spatial_scale),
        output_dim,
        group_size,
    )
    return [output.numpy()]


def _reference_clamp_floor(value: torch.Tensor, upper: int) -> torch.Tensor:
    """Clamp then floor a reference boundary without reusing Golden helpers."""
    positive = value > 0.0
    at_upper = value >= float(upper)
    converted = torch.floor(
        torch.where(positive & ~at_upper, value, torch.zeros_like(value))
    ).to(torch.long)
    converted = torch.clamp(converted, max=upper)
    return torch.where(
        positive,
        torch.where(at_upper, torch.full_like(converted, upper), converted),
        torch.zeros_like(converted),
    )


def _reference_clamp_ceil(value: torch.Tensor, upper: int) -> torch.Tensor:
    """Clamp then ceil a reference boundary without reusing Golden helpers."""
    positive = value > 0.0
    at_upper = value >= float(upper)
    converted = torch.ceil(
        torch.where(positive & ~at_upper, value, torch.zeros_like(value))
    ).to(torch.long)
    converted = torch.clamp(converted, max=upper)
    return torch.where(
        positive,
        torch.where(at_upper, torch.full_like(converted, upper), converted),
        torch.zeros_like(converted),
    )


def _reference_roi_bins(
    rois: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    spatial_scale: float,
    group_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Validate RoIs and calculate the independent reference bin boundaries."""
    rois_f32 = rois.to(torch.float32)
    rounded = torch.round(rois_f32)
    batch_ids = rounded[:, 0, :].to(torch.long)
    valid_rois = (
        torch.all(torch.isfinite(rois_f32), dim=1)
        & torch.all(rois_f32[:, 1:5, :] >= 0, dim=1)
        & (batch_ids >= 0)
        & (batch_ids < batch_size)
    )
    safe_rois = torch.where(
        valid_rois.unsqueeze(1), rois_f32, torch.zeros_like(rois_f32)
    )
    rounded = torch.round(safe_rois)
    batch_ids = rounded[:, 0, :].to(torch.long)
    scale = torch.as_tensor(spatial_scale, dtype=torch.float32, device=rois.device)
    start_w = rounded[:, 1, :] * scale
    start_h = rounded[:, 2, :] * scale
    end_w = (rounded[:, 3, :] + 1.0) * scale
    end_h = (rounded[:, 4, :] + 1.0) * scale
    roi_w = torch.maximum(
        end_w - start_w, torch.as_tensor(0.1, dtype=torch.float32, device=rois.device)
    )
    roi_h = torch.maximum(
        end_h - start_h, torch.as_tensor(0.1, dtype=torch.float32, device=rois.device)
    )
    grid = torch.arange(group_size, dtype=torch.float32, device=rois.device)
    h_start = _reference_clamp_floor(
        start_h.unsqueeze(-1) + grid * (roi_h / group_size).unsqueeze(-1), height
    )
    h_end = _reference_clamp_ceil(
        start_h.unsqueeze(-1) + (grid + 1.0) * (roi_h / group_size).unsqueeze(-1),
        height,
    )
    w_start = _reference_clamp_floor(
        start_w.unsqueeze(-1) + grid * (roi_w / group_size).unsqueeze(-1), width
    )
    w_end = _reference_clamp_ceil(
        start_w.unsqueeze(-1) + (grid + 1.0) * (roi_w / group_size).unsqueeze(-1), width
    )
    return valid_rois, batch_ids, h_start, h_end, w_start, w_end


def _reference_region_layout(
    batch_ids: torch.Tensor,
    h_start: torch.Tensor,
    h_end: torch.Tensor,
    w_start: torch.Tensor,
    w_end: torch.Tensor,
    *,
    batch_size: int,
    roi_count: int,
    channels: int,
    height: int,
    width: int,
    group_size: int,
    total_outputs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map output positions to source region widths, areas, and base offsets."""
    group_area = group_size * group_size
    flat_batches = batch_ids.reshape(-1)
    flat_h_start = h_start.reshape(batch_size * roi_count, group_size)
    flat_h_end = h_end.reshape(batch_size * roi_count, group_size)
    flat_w_start = w_start.reshape(batch_size * roi_count, group_size)
    flat_w_end = w_end.reshape(batch_size * roi_count, group_size)
    output_ids = torch.arange(total_outputs, dtype=torch.long, device=batch_ids.device)
    roi_slots = torch.div(output_ids, channels, rounding_mode="floor")
    channel_ids = output_ids - roi_slots * channels
    bin_ids = channel_ids % group_area
    ph = torch.div(bin_ids, group_size, rounding_mode="floor")
    pw = bin_ids - ph * group_size
    source_batch = flat_batches.index_select(0, roi_slots)
    hs = flat_h_start[roi_slots, ph]
    he = flat_h_end[roi_slots, ph]
    ws = flat_w_start[roi_slots, pw]
    we = flat_w_end[roi_slots, pw]
    region_width = we - ws
    area = (he - hs) * region_width
    base = ((source_batch * channels + channel_ids) * height + hs) * width + ws
    return roi_slots, region_width, area, base


def _reference_reduce_regions(
    x: torch.Tensor,
    region_width: torch.Tensor,
    area: torch.Tensor,
    base: torch.Tensor,
    *,
    feature_width: int,
) -> torch.Tensor:
    """Reduce independent reference regions with the NPU float32 semantics."""
    total_outputs = area.numel()
    flat_x = x.reshape(-1)
    valid_area = area > 0
    area_f32 = area.to(torch.float32).clamp_min(1.0)
    safe_term = (
        torch.as_tensor(np.finfo(np.float32).max, dtype=torch.float32, device=x.device)
        / area_f32
    )
    sums = torch.zeros(total_outputs, dtype=torch.float32, device=x.device)
    compensation = torch.zeros_like(sums)
    scaled = torch.zeros(total_outputs, dtype=torch.bool, device=x.device)
    has_nan = torch.zeros_like(scaled)
    has_pos_inf = torch.zeros_like(scaled)
    has_neg_inf = torch.zeros_like(scaled)
    max_area = int(area.max().item())
    for offset in range(max_area):
        offset_tensor = torch.as_tensor(offset, dtype=torch.long, device=x.device)
        valid = valid_area & (offset_tensor < area)
        rows = torch.div(
            torch.full_like(region_width, offset),
            region_width.clamp_min(1),
            rounding_mode="floor",
        )
        cols = offset - rows * region_width.clamp_min(1)
        indices = (base + rows * feature_width + cols).clamp_(0, flat_x.numel() - 1)
        values = flat_x.index_select(0, indices)
        finite = torch.isfinite(values)
        has_nan |= valid & torch.isnan(values)
        has_pos_inf |= valid & torch.isposinf(values)
        has_neg_inf |= valid & torch.isneginf(values)
        trigger = (
            valid & finite & ~scaled & ((values >= safe_term) | (values <= -safe_term))
        )
        sums = torch.where(trigger, sums / area_f32, sums)
        compensation = torch.where(trigger, compensation / area_f32, compensation)
        scaled |= trigger
        terms = torch.where(valid & finite, values, torch.zeros_like(values))
        terms = torch.where(scaled, terms / area_f32, terms)
        updated = sums + terms
        correction = torch.where(
            torch.abs(sums) >= torch.abs(terms),
            (sums - updated) + terms,
            (terms - updated) + sums,
        )
        compensation += correction
        sums = updated
    finite_result = torch.where(
        scaled, sums + compensation, (sums + compensation) / area_f32
    )
    result = torch.where(valid_area, finite_result, torch.zeros_like(finite_result))
    conflicting_inf = has_pos_inf & has_neg_inf
    result = torch.where(
        has_nan | conflicting_inf, torch.full_like(result, float("nan")), result
    )
    result = torch.where(
        has_pos_inf & ~has_neg_inf & ~has_nan,
        torch.full_like(result, float("inf")),
        result,
    )
    result = torch.where(
        has_neg_inf & ~has_pos_inf & ~has_nan,
        torch.full_like(result, -float("inf")),
        result,
    )
    return result


def _torch_device_reference_core(
    x: torch.Tensor,
    rois: torch.Tensor,
    spatial_scale: float,
    output_dim: int,
    group_size: int,
) -> torch.Tensor:
    """Independent Torch reference matching the NPU's float32 reduction path."""
    n, channels, height, width = x.shape
    roi_count = rois.shape[2]
    total_outputs = n * roi_count * channels
    if total_outputs == 0:
        return x.new_zeros((n * roi_count, output_dim, group_size, group_size))
    valid_rois, batch_ids, h_start, h_end, w_start, w_end = _reference_roi_bins(
        rois,
        batch_size=n,
        height=height,
        width=width,
        spatial_scale=spatial_scale,
        group_size=group_size,
    )
    roi_slots, region_width, area, base = _reference_region_layout(
        batch_ids,
        h_start,
        h_end,
        w_start,
        w_end,
        batch_size=n,
        roi_count=roi_count,
        channels=channels,
        height=height,
        width=width,
        group_size=group_size,
        total_outputs=total_outputs,
    )
    result = _reference_reduce_regions(x, region_width, area, base, feature_width=width)
    result = torch.where(
        valid_rois.reshape(-1).index_select(0, roi_slots),
        result,
        torch.zeros_like(result),
    )
    return result.to(x.dtype).reshape(n * roi_count, output_dim, group_size, group_size)


class PsroiPoolingV2TestSpec:
    """Kernel/GEIR TestSpec for psroi_pooling_v2."""

    class TorchDeviceReferenceImpl:
        """Independent device reference matching the NPU reduction semantics."""

        def __init__(
            self,
            *,
            spatial_scale,
            output_dim,
            group_size,
            **kwargs,
        ):
            del kwargs
            self.spatial_scale = float(spatial_scale)
            self.output_dim = int(output_dim)
            self.group_size = int(group_size)

        def __call__(self, x, rois, **kwargs):
            """Run an independent Torch implementation on the provider device."""
            del kwargs
            return [
                _torch_device_reference_core(
                    x,
                    rois,
                    self.spatial_scale,
                    self.output_dim,
                    self.group_size,
                )
            ]

    golden = staticmethod(psroi_pooling_v2_golden)

    third_party = {
        "torch": TorchDeviceReferenceImpl,
    }

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L0"},
        "float32": {"standard": "cross_check", "level": "L0"},
    }
