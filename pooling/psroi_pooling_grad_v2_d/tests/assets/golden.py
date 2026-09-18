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

"""Independent NumPy Golden for ``psroi_pooling_grad_v2_d``."""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch


__spec__ = {
    "psroi_pooling_grad_v2_d": "PSROIPoolingGradV2DTestSpec",
}


def golden_ps_roi_pooling_grad_v2(
    x,
    rois,
    spatial_scale,
    output_dim,
    group_size,
    input_size,
    **kwargs,
):
    """Golden function for ps_roi_pooling_grad_v2.

    Delegates to the shared Torch computation core so that the local
    golden and the remote third-party reference use identical logic.
    Tensor inputs are NumPy arrays (Kernel Golden contract).
    """
    del kwargs
    x = np.asarray(x)
    rois = np.asarray(rois)
    if x.dtype not in (np.float16, np.float32, np.float64) or rois.dtype != x.dtype:
        raise TypeError("x/rois must have the same float16/float32/float64 dtype")
    if x.ndim != 4 or rois.ndim != 3 or rois.shape[1] != 5:
        raise ValueError("invalid rank/rois shape")

    torch_x = torch.from_numpy(np.ascontiguousarray(x))
    torch_rois = torch.from_numpy(np.ascontiguousarray(rois))
    output = _ps_roi_pooling_grad_v2_torch_core(
        torch_x,
        torch_rois,
        spatial_scale,
        output_dim,
        group_size,
        input_size,
        coord_dtype=torch.float64,
    )
    return [output.numpy()]


def _ps_roi_pooling_grad_v2_torch_core(
    x,
    rois,
    spatial_scale,
    output_dim,
    group_size,
    input_size,
    coord_dtype=None,
):
    """Compute ps_roi_pooling_grad_v2 with vectorized Torch operations.

    Replaces the 5-level Python loop of the reference with a (ph, pw) loop
    plus chunked vectorized scatter-add over (batch, r_count, output_dim).
    ``x`` and ``rois`` are already-validated Torch tensors on the same device.

    ``coord_dtype`` controls the precision used for ROI coordinate computation
    (sw_f/sh_f/ew_f/eh_f, bin_w/bin_h, hs_f/he_f/ws_f/we_f). The CPU golden
    passes ``torch.float64`` for a high-precision reference; third-party device
    references should pass ``torch.float32`` (or leave None when calc_dtype is
    already fp32) to match the device operator's coordinate math.
    """
    batch, _, r_count = rois.shape
    group = int(group_size)
    height, width = int(input_size[0]), int(input_size[1])
    scale = float(spatial_scale)
    output_dim = int(output_dim)

    orig_dtype = x.dtype
    calc_dtype = torch.float32 if orig_dtype == torch.float16 else orig_dtype
    x = x.to(calc_dtype)
    rois = rois.to(calc_dtype)

    if coord_dtype is None:
        coord_dtype = torch.float64

    y = torch.zeros(
        (batch, output_dim * group * group, height, width),
        dtype=calc_dtype,
        device=x.device,
    )
    if group == 0 or height == 0 or width == 0 or r_count == 0 or output_dim == 0:
        return y.to(orig_dtype)

    br = batch * r_count
    coord = coord_dtype

    rounded = torch.round(rois)
    batch_ids = rounded[:, 0, :].to(torch.long).reshape(br)
    sw_i = rounded[:, 1, :].to(torch.long).reshape(br)
    sh_i = rounded[:, 2, :].to(torch.long).reshape(br)
    ew_i = (rounded[:, 3, :].to(torch.long) + 1).reshape(br)
    eh_i = (rounded[:, 4, :].to(torch.long) + 1).reshape(br)

    sw_f = sw_i.to(coord) * scale
    sh_f = sh_i.to(coord) * scale
    ew_f = ew_i.to(coord) * scale
    eh_f = eh_i.to(coord) * scale
    roi_w = (ew_f - sw_f).clamp(min=0.1)
    roi_h = (eh_f - sh_f).clamp(min=0.1)
    bin_w = roi_w / group
    bin_h = roi_h / group

    h_idx = torch.arange(height, device=x.device)
    w_idx = torch.arange(width, device=x.device)

    elem_per_q = max(output_dim * height * width, 1)
    chunk = max(1, 50_000_000 // elem_per_q)

    y_view = y.view(batch, output_dim, group * group, height, width)

    for ph in range(group):
        hs_f = ph * bin_h + sh_f
        he_f = (ph + 1) * bin_h + sh_f
        hs = hs_f.floor().to(torch.long).clamp(0, height)
        he = he_f.ceil().to(torch.long).clamp(0, height)
        for pw in range(group):
            ws_f = pw * bin_w + sw_f
            we_f = (pw + 1) * bin_w + sw_f
            ws = ws_f.floor().to(torch.long).clamp(0, width)
            we = we_f.ceil().to(torch.long).clamp(0, width)
            ch_base = ph * group + pw
            x_slice = x[:, :, ph, pw]

            for q0 in range(0, br, chunk):
                q1 = min(q0 + chunk, br)
                h_m = (h_idx[None, :] >= hs[q0:q1, None]) & (
                    h_idx[None, :] < he[q0:q1, None]
                )
                w_m = (w_idx[None, :] >= ws[q0:q1, None]) & (
                    w_idx[None, :] < we[q0:q1, None]
                )
                mask = h_m[:, :, None] & w_m[:, None, :]
                area = (he[q0:q1] - hs[q0:q1]) * (we[q0:q1] - ws[q0:q1])
                area_f = area.to(calc_dtype)
                inv_area = torch.where(
                    area_f > 0,
                    1.0 / area_f.clamp(min=1),
                    torch.zeros_like(area_f),
                )
                vals = x_slice[q0:q1] * inv_area[:, None]
                t = vals[:, :, None, None] * mask[:, None, :, :].to(calc_dtype)
                y_view[:, :, ch_base, :, :].index_add_(0, batch_ids[q0:q1], t)

    return y.to(orig_dtype)


class PSROIPoolingGradV2DTestSpec:
    """Kernel/GEIR TestSpec for ``psroi_pooling_grad_v2_d``."""

    @staticmethod
    def golden(
        x,
        rois,
        spatial_scale,
        output_dim,
        group_size,
        input_size,
        **kwargs,
    ):
        return [
            golden_ps_roi_pooling_grad_v2(
                x,
                rois,
                spatial_scale,
                output_dim,
                group_size,
                input_size,
                **kwargs,
            )
        ]

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }

    class TorchDeviceReferenceImpl:
        """Device-side PyTorch reference implementation for third-party cross-check."""

        def __init__(
            self,
            *,
            spatial_scale,
            output_dim,
            group_size,
            input_size,
            **kwargs,
        ):
            del kwargs
            self.spatial_scale = float(spatial_scale)
            self.output_dim = int(output_dim)
            self.group_size = int(group_size)
            self.input_size = (int(input_size[0]), int(input_size[1]))

        def __call__(self, x, rois, **kwargs):
            """Adapt provider tensors to the shared Torch computation core."""
            del kwargs
            rois = rois.to(x.dtype)
            return [
                _ps_roi_pooling_grad_v2_torch_core(
                    x,
                    rois,
                    self.spatial_scale,
                    self.output_dim,
                    self.group_size,
                    self.input_size,
                    coord_dtype=torch.float32,
                )
            ]

    third_party = {
        "torch": TorchDeviceReferenceImpl,
    }
