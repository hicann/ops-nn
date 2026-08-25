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

"""TTK Golden implementation for ``psroi_pooling_v2``.

The Kernel Golden contract uses NumPy arrays, while the remote provider uses
device Torch tensors. Both adapters call one vectorized Torch Core following
the canndev TBE V2 formula.
"""

import math
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch


__spec__ = {
    "psroi_pooling_v2": "PsroiPoolingV2TestSpec",
}


def _psroi_pooling_v2_torch_core(
    x: torch.Tensor,
    rois: torch.Tensor,
    spatial_scale,
    output_dim,
    group_size,
):
    """Compute PSROIPoolingV2 with vectorized device-side Torch operations.

    ``x`` and ``rois`` are already-validated Torch tensors on the same device.
    This function is the single computation truth shared by the Kernel Golden
    wrapper and the remote Torch provider implementation.
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
    rounded = torch.round(rois.to(torch.float64))
    batch_ids = rounded[:, 0, :].to(torch.long)
    roi_start_w = rounded[:, 1, :] * spatial_scale
    roi_start_h = rounded[:, 2, :] * spatial_scale
    roi_end_w = (rounded[:, 3, :] + 1.0) * spatial_scale
    roi_end_h = (rounded[:, 4, :] + 1.0) * spatial_scale
    roi_w = torch.clamp_min(roi_end_w - roi_start_w, 0.1)
    roi_h = torch.clamp_min(roi_end_h - roi_start_h, 0.1)
    bin_w = roi_w / group_size
    bin_h = roi_h / group_size

    grid = torch.arange(
        group_size,
        dtype=torch.float64,
        device=x.device,
    )
    hstart = (
        torch.floor(roi_start_h.unsqueeze(-1) + grid * bin_h.unsqueeze(-1))
        .clamp(0, h)
        .to(torch.long)
    )
    hend = (
        torch.ceil(roi_start_h.unsqueeze(-1) + (grid + 1.0) * bin_h.unsqueeze(-1))
        .clamp(0, h)
        .to(torch.long)
    )
    wstart = (
        torch.floor(roi_start_w.unsqueeze(-1) + grid * bin_w.unsqueeze(-1))
        .clamp(0, w)
        .to(torch.long)
    )
    wend = (
        torch.ceil(roi_start_w.unsqueeze(-1) + (grid + 1.0) * bin_w.unsqueeze(-1))
        .clamp(0, w)
        .to(torch.long)
    )

    prefix = (
        torch.nn.functional.pad(
            x.to(acc_dtype),
            (1, 0, 1, 0),
        )
        .cumsum(dim=-1)
        .cumsum(dim=-2)
    )

    hstart = hstart.unsqueeze(-1).expand(n, r_count, group_size, group_size)
    hend = hend.unsqueeze(-1).expand(n, r_count, group_size, group_size)
    wstart = wstart.unsqueeze(-2).expand(n, r_count, group_size, group_size)
    wend = wend.unsqueeze(-2).expand(n, r_count, group_size, group_size)
    channel_bins = torch.arange(
        group_size * group_size,
        dtype=torch.long,
        device=x.device,
    ).reshape(group_size, group_size)
    channels = (
        torch.arange(output_dim, dtype=torch.long, device=x.device).reshape(
            output_dim, 1, 1
        )
        * (group_size * group_size)
        + channel_bins
    )

    out_shape = (n, r_count, output_dim, group_size, group_size)
    batch_index = batch_ids.reshape(n, r_count, 1, 1, 1).expand(out_shape)
    channel_index = channels.reshape(1, 1, output_dim, group_size, group_size).expand(
        out_shape
    )
    hs = hstart.unsqueeze(2).expand(out_shape)
    he = hend.unsqueeze(2).expand(out_shape)
    ws = wstart.unsqueeze(2).expand(out_shape)
    we = wend.unsqueeze(2).expand(out_shape)

    region_sum = (
        prefix[batch_index, channel_index, he, we]
        - prefix[batch_index, channel_index, hs, we]
        - prefix[batch_index, channel_index, he, ws]
        + prefix[batch_index, channel_index, hs, ws]
    )
    area = (he - hs) * (we - ws)
    output = torch.where(
        area > 0,
        region_sum / area.clamp_min(1).to(acc_dtype),
        torch.zeros((), dtype=acc_dtype, device=x.device),
    )
    return output.reshape(
        n * r_count,
        output_dim,
        group_size,
        group_size,
    ).to(x.dtype)


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
        **kwargs: TTK metadata such as input/output dtypes and formats.

    Returns:
        A one-element list containing the NumPy output array with shape
        ``[N * R, output_dim, group_size, group_size]``.
    """
    del kwargs
    x = np.asarray(x)
    rois = np.asarray(rois)

    supported = (np.dtype(np.float16), np.dtype(np.float32))
    if x.dtype not in supported or rois.dtype != x.dtype:
        raise TypeError("x/rois must have the same float16 or float32 dtype")
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
    if not torch.isfinite(torch_rois).all().item():
        raise ValueError("rois must not contain NaN or Inf")
    if torch.any(torch_rois[:, 1:5, :] < 0).item():
        raise ValueError("RoI coordinates must be non-negative")
    rounded_batch_ids = torch.round(torch_rois[:, 0, :]).to(torch.int64)
    if torch.any((rounded_batch_ids < 0) | (rounded_batch_ids >= n)).item():
        raise ValueError("rounded RoI batch_id is out of range")

    output = _psroi_pooling_v2_torch_core(
        torch_x,
        torch_rois,
        float(spatial_scale),
        output_dim,
        group_size,
    )
    return [output.numpy()]


class PsroiPoolingV2TestSpec:
    """Kernel/GEIR TestSpec for psroi_pooling_v2."""

    class TorchDeviceReferenceImpl:
        """Vectorized device-side PyTorch Core reference implementation."""

        def __init__(
            self,
            *,
            spatial_scale,
            output_dim,
            group_size,
            **kwargs,
        ):
            del kwargs
            if not isinstance(spatial_scale, (int, float)):
                raise TypeError("spatial_scale must be a Python scalar")
            if (
                spatial_scale <= 0
                or spatial_scale != spatial_scale
                or abs(spatial_scale) == float("inf")
            ):
                raise ValueError("spatial_scale must be positive and finite")
            if int(output_dim) != output_dim or output_dim <= 0:
                raise ValueError("output_dim must be a positive integer")
            if int(group_size) != group_size or not 1 <= group_size < 128:
                raise ValueError("group_size must be an integer in [1, 127]")
            self.spatial_scale = float(spatial_scale)
            self.output_dim = int(output_dim)
            self.group_size = int(group_size)

        def __call__(self, x, rois, **kwargs):
            """Adapt provider tensors to the shared Torch computation core."""
            del kwargs
            if not isinstance(x, torch.Tensor) or not isinstance(rois, torch.Tensor):
                raise TypeError("TTK third_party inputs x/rois must be torch.Tensor")
            if x.device != rois.device:
                raise ValueError("x and rois must be on the same provider device")
            if x.dtype not in (torch.float16, torch.float32) or rois.dtype != x.dtype:
                raise TypeError("x/rois must have the same float16 or float32 dtype")
            if x.ndim != 4:
                raise ValueError("x must be a rank-4 ND tensor [N,C,H,W]")
            if rois.ndim != 3 or rois.shape[1] != 5:
                raise ValueError("rois must have shape [N,5,R]")

            n, c, h, w = x.shape
            if rois.shape[0] != n:
                raise ValueError("rois.shape[0] must equal x.shape[0]")
            if h <= 0 or w <= 0:
                raise ValueError("H and W must be greater than zero")
            output_dim = self.output_dim
            group_size = self.group_size
            if c != output_dim * group_size * group_size:
                raise ValueError("C must equal output_dim*group_size^2")
            return [
                _psroi_pooling_v2_torch_core(
                    x,
                    rois,
                    self.spatial_scale,
                    output_dim,
                    group_size,
                )
            ]

    golden = staticmethod(psroi_pooling_v2_golden)

    third_party = {
        "torch": TorchDeviceReferenceImpl,
    }

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
