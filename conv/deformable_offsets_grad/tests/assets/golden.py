#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent Torch references for the Kernel/GEIR DeformableOffsetsGrad contract.

The primary golden evaluates the analytic bilinear derivative on CPU. The
third-party reference differentiates a separately constructed, vectorized
bilinear sampler on the input device. Neither calls the NPU operator or the
other reference. NumPy is only used at the Kernel input/output boundary.

Tensor storage is NHWC. ``data_format`` selects the H/W entries of strides and
dilations; both NHWC and NCHW attribute encodings are accepted. Offsets use
[component(x, y, mask), group, kh, kw] channel order. The current operator only
supports modulated=True. At coordinates outside the open interval
(-1, height) x (-1, width), both input and offset gradients are zero. The mask
gradient is the sampled input times the upstream gradient, without a mask
factor. Both references evaluate in float64 to retain precision when channel
derivatives cancel, then cast outputs to the declared dtype. This is a
numerical reference, not an emulation of atomic-add order.
Integer coordinates and fractional offsets are kept separate when selecting
the interpolation cell, so tiny offsets cannot round onto a different cell.
Non-finite offsets (including masks) are rejected by both references.
"""

import numpy as np
import torch


__input__ = {"kernel": {"deformable_offsets_grad": "deformable_offsets_grad_input"}}
__golden__ = {"kernel": {"deformable_offsets_grad": "deformable_offsets_grad_golden"}}
__spec__ = {"deformable_offsets_grad": "DeformableOffsetsGradKernelSpec"}


def _to_torch(arr):
    """Bridge NumPy, including ml_dtypes.bfloat16, to a CPU float64 tensor."""
    return torch.from_numpy(np.asarray(arr).astype(np.float64, copy=False))


def _to_numpy(tensor, dtype):
    """Bridge back to the declared Kernel output dtype."""
    array = tensor.detach().cpu().numpy()
    name = dtype if isinstance(dtype, str) else np.dtype(dtype).name
    if name == "bfloat16":
        import ml_dtypes

        return array.astype(ml_dtypes.bfloat16)
    return array.astype(np.dtype(name), copy=False)


def _torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    name = dtype if isinstance(dtype, str) else np.dtype(dtype).name
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def _hw_components(seq, data_format):
    if data_format not in ("NHWC", "NCHW"):
        raise ValueError("data_format must be NHWC or NCHW")
    if len(seq) != 4:
        raise ValueError("strides and dilations must have four entries")
    return int(seq[data_format.index("H")]), int(seq[data_format.index("W")])


def _validate_inputs(grad, x, offsets, ksize, groups, modulated):
    """Validate the supported contract without restricting finite offset range."""
    if not modulated:
        raise ValueError("DeformableOffsetsGrad currently requires modulated=True")
    if len(ksize) != 2 or min(ksize) <= 0:
        raise ValueError("ksize must contain two positive dimensions")
    if x.ndim != 4 or grad.ndim != 4 or offsets.ndim != 4:
        raise ValueError("grad, x and offsets must be NHWC tensors")
    n, _, _, channels = x.shape
    if groups <= 0 or channels % groups:
        raise ValueError("deformable_groups must divide the input channels")
    kh, kw = ksize
    if offsets.shape[0] != n or offsets.shape[-1] != 3 * groups * kh * kw:
        raise ValueError("offsets must have 3 * groups * kH * kW channels")
    expected = (n, offsets.shape[1] * kh, offsets.shape[2] * kw, channels)
    if tuple(grad.shape) != expected:
        raise ValueError(f"grad shape {tuple(grad.shape)} must be {expected}")
    if not bool(torch.isfinite(offsets).all()):
        raise ValueError("offsets, including masks, must be finite")


def _axis_weights(base, offset):
    """CPU analytic cell and weights, without rounding base + offset."""
    numerator, denominator = offset.item().as_integer_ratio()
    whole, remainder = divmod(numerator, denominator)
    # Compute both weights independently: 1 - rounded(1 - epsilon) loses epsilon.
    return (
        base + whole,
        (denominator - remainder) / denominator,
        remainder / denominator,
    )


def deformable_offsets_grad_golden(
    grad,
    x,
    offsets,
    strides,
    pads,
    ksize,
    dilations=(1, 1, 1, 1),
    data_format="NHWC",
    deformable_groups=1,
    modulated=True,
    **kwargs,
):
    """CPU analytic reference; return NumPy [grad_x, grad_offsets]."""
    out_dtypes = kwargs.get("output_dtypes")
    gx_dtype = out_dtypes[0] if out_dtypes else x.dtype
    goff_dtype = out_dtypes[1] if out_dtypes and len(out_dtypes) > 1 else offsets.dtype
    grad_f, x_f, off_f = (_to_torch(value) for value in (grad, x, offsets))
    k_h, k_w = int(ksize[0]), int(ksize[1])
    groups = int(deformable_groups)
    _validate_inputs(grad_f, x_f, off_f, (k_h, k_w), groups, modulated)
    stride_h, stride_w = _hw_components(strides, data_format)
    dil_h, dil_w = _hw_components(dilations, data_format)
    pad_u, _, pad_l, _ = (int(p) for p in pads)
    n, in_h, in_w, in_c = x_f.shape
    _, out_h, out_w, _ = off_f.shape
    cpg = in_c // groups
    group_stride = k_h * k_w
    comp_stride = groups * group_stride
    grad_x = torch.zeros_like(x_f)
    grad_off = torch.zeros_like(off_f)

    for bn in range(n):
        for oh in range(out_h):
            for ow in range(out_w):
                for kh in range(k_h):
                    for kw in range(k_w):
                        for group in range(groups):
                            channel = group * group_stride + kh * k_w + kw
                            off_x = off_f[bn, oh, ow, channel]
                            off_y = off_f[bn, oh, ow, comp_stride + channel]
                            mask = off_f[bn, oh, ow, 2 * comp_stride + channel]
                            h0, hh, lh = _axis_weights(
                                oh * stride_h - pad_u + kh * dil_h, off_y
                            )
                            w0, hw, lw = _axis_weights(
                                ow * stride_w - pad_l + kw * dil_w, off_x
                            )
                            # The lower neighbour plus a nonzero upper weight
                            # distinguishes -1 + epsilon from the excluded -1.
                            if not (
                                -1 <= h0 < in_h
                                and -1 <= w0 < in_w
                                and (h0 != -1 or lh > 0)
                                and (w0 != -1 or lw > 0)
                            ):
                                continue
                            gc = slice(group * cpg, (group + 1) * cpg)
                            g_vec = grad_f[bn, oh * k_h + kh, ow * k_w + kw, gc]
                            g_h = x_f.new_zeros(())
                            g_w = x_f.new_zeros(())
                            g_mask = x_f.new_zeros(())
                            corners = (
                                (h0, w0, hh * hw, -hw, -hh),
                                (h0, w0 + 1, hh * lw, -lw, hh),
                                (h0 + 1, w0, lh * hw, hw, -lh),
                                (h0 + 1, w0 + 1, lh * lw, lw, lh),
                            )
                            for hi, wi, weight, dh, dw in corners:
                                if not (0 <= hi < in_h and 0 <= wi < in_w):
                                    continue
                                value = x_f[bn, hi, wi, gc]
                                grad_x[bn, hi, wi, gc] += g_vec * (mask * weight)
                                g_h += (g_vec * value * dh).sum()
                                g_w += (g_vec * value * dw).sum()
                                g_mask += (g_vec * value * weight).sum()
                            grad_off[bn, oh, ow, channel] = mask * g_w
                            grad_off[bn, oh, ow, comp_stride + channel] = mask * g_h
                            grad_off[bn, oh, ow, 2 * comp_stride + channel] = g_mask
    return [_to_numpy(grad_x, gx_dtype), _to_numpy(grad_off, goff_dtype)]


def deformable_offsets_grad_input(
    grad,
    x,
    offsets,
    strides,
    pads,
    ksize,
    dilations=(1, 1, 1, 1),
    data_format="NHWC",
    deformable_groups=1,
    modulated=True,
    **kwargs,
):
    """Preserve the existing small fractional offsets for interior cases."""
    if "interior" not in (kwargs.get("testcase_name", "") or ""):
        return [grad, x, offsets]
    per = int(deformable_groups) * int(ksize[0]) * int(ksize[1])
    new_off = _to_torch(offsets).clone()
    ramp = 0.25 + 0.20 * torch.arange(
        per, device=new_off.device, dtype=torch.float32
    ) / max(per - 1, 1)
    new_off[..., :per] = ramp
    new_off[..., per : 2 * per] = ramp
    new_off[..., 2 * per : 3 * per] = 1.0
    return [grad, x, _to_numpy(new_off, offsets.dtype)]


class DeformableOffsetsGradTorchCompetitor:
    """Independent device-local sampler differentiated by Torch autograd.

    The forward construction uses gather and bilinear weights; autograd supplies
    both offset derivatives and accumulation for repeated input samples. It does
    not reuse the analytic backward above or call a deformable-convolution DUT.
    """

    def __init__(
        self,
        strides,
        pads,
        ksize,
        dilations=(1, 1, 1, 1),
        data_format="NHWC",
        deformable_groups=1,
        modulated=True,
        **kwargs,
    ):
        self.stride_h, self.stride_w = _hw_components(strides, data_format)
        self.dil_h, self.dil_w = _hw_components(dilations, data_format)
        self.pad_u, self.pad_l = int(pads[0]), int(pads[2])
        self.k_h, self.k_w = int(ksize[0]), int(ksize[1])
        self.groups = int(deformable_groups)
        self.modulated = modulated
        self.output_dtypes = kwargs.get("output_dtypes")

    def __call__(self, grad, x, offsets, **kwargs):
        if grad.device != x.device or offsets.device != x.device:
            raise ValueError("grad, x and offsets must be on the same device")
        out_dtypes = kwargs.get("output_dtypes") or self.output_dtypes
        gx_dtype = _torch_dtype(out_dtypes[0]) if out_dtypes else x.dtype
        goff_dtype = (
            _torch_dtype(out_dtypes[1])
            if out_dtypes and len(out_dtypes) > 1
            else offsets.dtype
        )
        # The harness may call references under no_grad/inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            data = x.detach().to(torch.float64).clone().requires_grad_(True)
            off = offsets.detach().to(torch.float64).clone().requires_grad_(True)
            upstream = grad.detach().to(torch.float64)
            _validate_inputs(
                upstream, data, off, (self.k_h, self.k_w), self.groups, self.modulated
            )
            n, in_h, in_w, channels = data.shape
            if in_h == 0 or in_w == 0:
                # Preserve the analytic zero-padding result without indexing an empty image.
                return [
                    torch.zeros_like(data).to(gx_dtype),
                    torch.zeros_like(off).to(goff_dtype),
                ]
            _, out_h, out_w, _ = off.shape
            k_h, k_w = self.k_h, self.k_w
            cpg = channels // self.groups
            block = k_h * k_w
            component = self.groups * block
            device = data.device
            oh = torch.arange(out_h, device=device, dtype=torch.float64).view(
                1, out_h, 1, 1, 1
            )
            ow = torch.arange(out_w, device=device, dtype=torch.float64).view(
                1, 1, out_w, 1, 1
            )
            kh = torch.arange(k_h, device=device, dtype=torch.float64).view(
                1, 1, 1, k_h, 1
            )
            kw = torch.arange(k_w, device=device, dtype=torch.float64).view(
                1, 1, 1, 1, k_w
            )
            batch = torch.arange(n, device=device).view(n, 1, 1, 1, 1)
            samples = []
            query_shape = (n, out_h, out_w, k_h, k_w)
            for group in range(self.groups):
                start = group * block
                dx = off[..., start : start + block].reshape(query_shape)
                dy = off[..., component + start : component + start + block].reshape(
                    query_shape
                )
                mask = off[
                    ..., 2 * component + start : 2 * component + start + block
                ].reshape(query_shape)
                whole_h, whole_w = torch.trunc(dy), torch.trunc(dx)
                frac_h, frac_w = dy - whole_h, dx - whole_w
                negative_h, negative_w = frac_h < 0, frac_w < 0
                low_h = oh * self.stride_h - self.pad_u + kh * self.dil_h + whole_h
                low_w = ow * self.stride_w - self.pad_l + kw * self.dil_w + whole_w
                low_h = low_h - negative_h.to(torch.float64)
                low_w = low_w - negative_w.to(torch.float64)
                # Signed fractions preserve the small weight on either side
                # of a knot. At exact integers choose the right-hand cell.
                h_lower = torch.where(negative_h, -frac_h, 1.0 - frac_h)
                h_upper = torch.where(negative_h, 1.0 + frac_h, frac_h)
                w_lower = torch.where(negative_w, -frac_w, 1.0 - frac_w)
                w_upper = torch.where(negative_w, 1.0 + frac_w, frac_w)
                gate = (
                    (low_h >= -1)
                    & (low_h < in_h)
                    & ((low_h != -1) | (h_upper > 0))
                    & (low_w >= -1)
                    & (low_w < in_w)
                    & ((low_w != -1) | (w_upper > 0))
                )
                # A sentinel keeps far-out finite coordinates away from int overflow.
                h0 = torch.where(gate, low_h, torch.zeros_like(low_h)).to(torch.int64)
                w0 = torch.where(gate, low_w, torch.zeros_like(low_w)).to(torch.int64)
                gc = slice(group * cpg, (group + 1) * cpg)
                value = torch.zeros(
                    (*query_shape, cpg), device=device, dtype=torch.float64
                )
                for hi, h_weight in ((h0, h_lower), (h0 + 1, h_upper)):
                    for wi, w_weight in ((w0, w_lower), (w0 + 1, w_upper)):
                        valid = gate & (hi >= 0) & (hi < in_h) & (wi >= 0) & (wi < in_w)
                        corner = data[
                            batch, hi.clamp(0, in_h - 1), wi.clamp(0, in_w - 1), gc
                        ]
                        corner = torch.where(
                            valid.unsqueeze(-1), corner, torch.zeros_like(corner)
                        )
                        value = value + corner * (h_weight * w_weight).unsqueeze(-1)
                sampled = value * mask.unsqueeze(-1)
                samples.append(
                    torch.where(gate.unsqueeze(-1), sampled, torch.zeros_like(sampled))
                )
            sampled_output = torch.cat(samples, dim=-1)
            query_grad = upstream.reshape(n, out_h, k_h, out_w, k_w, channels).permute(
                0, 1, 3, 2, 4, 5
            )
            grad_x, grad_offsets = torch.autograd.grad(
                sampled_output,
                (data, off),
                grad_outputs=query_grad,
                create_graph=False,
                retain_graph=False,
            )
        return [grad_x.detach().to(gx_dtype), grad_offsets.detach().to(goff_dtype)]


class DeformableOffsetsGradKernelSpec:
    """Kernel/GEIR analytic golden with an independent autodiff third party."""

    golden = staticmethod(deformable_offsets_grad_golden)
    third_party = {"torch": DeformableOffsetsGradTorchCompetitor}
    customize_inputs = staticmethod(deformable_offsets_grad_input)
    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
