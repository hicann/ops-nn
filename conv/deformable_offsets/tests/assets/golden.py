#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden and independent torch reference for the DeformableOffsets forward operator.

Contract (``deformable_offsets_def.cpp`` / arch35 kernel ``deformable_offsets.h``):
  * NHWC tensors, offsets channel layout ``[component(0=x, 1=y, 2=mask), group, kh, kw]``;
  * sampled coordinate ``pointH = oh*strideH - padU + kh*dilH + offY`` (W analogous);
  * bilinear interpolation with per-neighbour out-of-bound -> 0 (``__floorf`` + 4 corners);
  * ``y[n, oh*kH+kh, ow*kW+kw, c] = mask * sum(weight_ij * x[n, hi, wj, c])``.
  Host tiling accepts only ``modulated=True``; the kernel always reads and
  multiplies the mask component. Both references enforce that supported domain.
  Low precision (fp16/bf16) is computed in fp32 and cast back to the output dtype,
  mirroring the kernel float accumulation.

Reference support boundary (FR-04): both references reject non-finite offsets,
including the mask component, rather than inventing a numeric answer. This is
not a claim that the device rejects those inputs. Finite coordinates outside
the image keep the existing per-neighbour zero-padding semantics, including
coordinates too large for a tensor integer conversion. Device behaviour at
those extremes requires separate validation.
"""

import numpy as np
import torch


__input__ = {"kernel": {"deformable_offsets": "deformable_offsets_input"}}
__golden__ = {"kernel": {"deformable_offsets": "deformable_offsets_golden"}}
__spec__ = {"deformable_offsets": "DeformableOffsetsKernelSpec"}


def _as_tensor(value, device=None):
    """Bridge containers without performing reference arithmetic in NumPy."""
    if torch.is_tensor(value):
        return (
            value.detach().to(device=device) if device is not None else value.detach()
        )
    array = np.ascontiguousarray(value)
    if array.dtype.name == "bfloat16":
        tensor = torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(array)
    return tensor.to(device=device) if device is not None else tensor


def _torch_dtype(value):
    if isinstance(value, torch.dtype):
        return value
    name = value if isinstance(value, str) else np.dtype(value).name
    name = name.removeprefix("torch.")
    supported = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        # TTK Promote requests this output storage; arithmetic remains fp32.
        "float64": torch.float64,
    }
    if name not in supported:
        raise ValueError(f"unsupported DeformableOffsets output dtype: {value!r}")
    return supported[name]


def _to_numpy(tensor, dst_dtype):
    """Cast in Torch; NumPy only views the final host storage."""
    tensor = (
        tensor.detach().to(device="cpu", dtype=_torch_dtype(dst_dtype)).contiguous()
    )
    if tensor.dtype == torch.bfloat16:
        import ml_dtypes  # noqa: F401  (register the NumPy storage dtype)

        return tensor.view(torch.uint16).numpy().view(np.dtype("bfloat16"))
    return tensor.numpy()


def _hw_components(seq, data_format):
    """Extract the (H, W) entries from a 4-length NHWC/NCHW attribute list."""
    seq = list(seq)
    h_idx = data_format.index("H")
    w_idx = data_format.index("W")
    return seq[h_idx], seq[w_idx]


def _resolve_output_dtype(default_dtype, **kwargs):
    out_dtypes = kwargs.get("output_dtypes")
    if out_dtypes is not None and len(out_dtypes):
        return _torch_dtype(out_dtypes[0])
    return default_dtype


def _validate_inputs(
    x, offsets, strides, pads, ksize, dilations, data_format, groups, modulated
):
    """Shared input/attribute checks, not a shared numerical reference."""
    if x.ndim != 4 or offsets.ndim != 4:
        raise ValueError("DeformableOffsets expects physical NHWC tensors of rank four")
    if data_format not in ("NHWC", "NCHW"):
        raise ValueError("data_format must be NHWC or NCHW for attribute indexing")
    if len(strides) != 4 or len(dilations) != 4 or len(pads) != 4 or len(ksize) != 2:
        raise ValueError(
            "strides/dilations/pads must have length four and ksize length two"
        )
    if not modulated:
        raise ValueError("DeformableOffsets host tiling supports only modulated=True")
    if not bool(torch.isfinite(offsets).all()):
        raise ValueError(
            "DeformableOffsets references reject non-finite offsets (including mask)"
        )
    n, _, _, in_c = x.shape
    off_n, out_h, out_w, off_c = offsets.shape
    k_h, k_w = (int(v) for v in ksize)
    stride_h, stride_w = (int(v) for v in _hw_components(strides, data_format))
    if min(in_c, out_h, out_w, k_h, k_w, stride_h, stride_w, groups) <= 0:
        raise ValueError(
            "output spatial sizes, channels, groups, strides and ksize must be positive"
        )
    if n != off_n:
        raise ValueError("offsets batch dimension does not match x")
    if off_c != 3 * groups * k_h * k_w or in_c % groups:
        raise ValueError(
            "offsets channels or input channels do not match deformable_groups/ksize"
        )


def deformable_offsets_golden(
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
    """Kernel golden for ``deformable_offsets``.

    Parameter names/order follow ``deformable_offsets_def.cpp`` (inputs then attrs).
    Inputs are ``numpy.ndarray``. Returns ``[y]``.
    """
    x_tensor = _as_tensor(x, device="cpu")
    out_dtype = _resolve_output_dtype(x_tensor.dtype, **kwargs)
    x_f = x_tensor.to(torch.float32)
    off_f = _as_tensor(offsets, device="cpu").to(torch.float32)
    groups = int(deformable_groups)
    _validate_inputs(
        x_f, off_f, strides, pads, ksize, dilations, data_format, groups, modulated
    )

    n, in_h, in_w, in_c = x_f.shape
    _, out_h, out_w, off_c = off_f.shape

    k_h, k_w = int(ksize[0]), int(ksize[1])
    stride_h, stride_w = _hw_components(strides, data_format)
    dil_h, dil_w = _hw_components(dilations, data_format)
    pad_u, pad_d, pad_l, pad_r = (int(p) for p in pads)
    stride_h, stride_w = int(stride_h), int(stride_w)
    dil_h, dil_w = int(dil_h), int(dil_w)
    cpg = in_c // groups  # channels per group

    comp_stride = groups * k_h * k_w  # stride between (x, y, mask) components
    group_stride = k_h * k_w

    y = x_f.new_zeros((n, out_h * k_h, out_w * k_w, in_c))

    for bn in range(n):
        for oh in range(out_h):
            for ow in range(out_w):
                for kh in range(k_h):
                    for kw in range(k_w):
                        base = kh * k_w + kw
                        for g in range(groups):
                            ch = g * group_stride + base
                            off_x = off_f[bn, oh, ow, 0 * comp_stride + ch]
                            off_y = off_f[bn, oh, ow, 1 * comp_stride + ch]
                            mask = off_f[bn, oh, ow, 2 * comp_stride + ch]

                            # Keep both additions in fp32, as in the kernel.
                            point_h = x_f.new_tensor(oh * stride_h - pad_u)
                            point_h = point_h + x_f.new_tensor(kh * dil_h) + off_y
                            point_w = x_f.new_tensor(ow * stride_w - pad_l)
                            point_w = point_w + x_f.new_tensor(kw * dil_w) + off_x

                            h0 = torch.floor(point_h)
                            w0 = torch.floor(point_w)
                            dh = point_h - h0
                            dw = point_w - w0

                            gc = slice(g * cpg, (g + 1) * cpg)
                            acc = x_f.new_zeros(cpg)
                            # Convert only in-bounds coordinates to integer indices.
                            # Finite far-outside coordinates therefore remain zero
                            # contributions without overflowing a tensor int64 cast.
                            for hh, wgt_h in ((h0, 1.0 - dh), (h0 + 1, dh)):
                                if hh < 0 or hh >= in_h:
                                    continue
                                for ww, wgt_w in ((w0, 1.0 - dw), (w0 + 1, dw)):
                                    if ww < 0 or ww >= in_w:
                                        continue
                                    value = x_f[bn, int(hh.item()), int(ww.item()), gc]
                                    acc = acc + (wgt_h * wgt_w) * value

                            out_row = oh * k_h + kh
                            out_col = ow * k_w + kw
                            y[bn, out_row, out_col, gc] = mask * acc

    return [_to_numpy(y, out_dtype)]


def deformable_offsets_input(
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
    """Custom input generator for special DeformableOffsets cases.

    Keyed on ``testcase_name``:
      * ``*zero_offset*`` -> set x/y offset components to 0 and mask to 1
        (degenerates to a regular gather at integer positions).
    Otherwise inputs are returned unchanged.
    """
    name = kwargs.get("testcase_name", "") or ""
    if "zero_offset" not in name:
        return [x, offsets]

    k_h, k_w = int(ksize[0]), int(ksize[1])
    groups = int(deformable_groups)
    comp_stride = groups * k_h * k_w

    new_off = np.array(offsets, copy=True)
    # component 0 (x offset) and 1 (y offset) -> 0
    new_off[..., 0 * comp_stride : 1 * comp_stride] = 0
    new_off[..., 1 * comp_stride : 2 * comp_stride] = 0
    # component 2 (mask) -> 1
    one = np.ones(1, dtype=new_off.dtype)[0]
    new_off[..., 2 * comp_stride : 3 * comp_stride] = one
    return [x, new_off]


class DeformableOffsetsTorchCompetitor:
    """Independent torch reference for the Kernel/GEIR ``deformable_offsets`` entry.

    Receives ``torch.Tensor`` inputs (TTK converts numpy -> torch for the kernel
    path), computes the bilinear sampling with pure torch tensor ops on the
    configured device (CPU/GPU), and returns ``[y]``.  It never calls the NPU DUT
    nor any deformable-convolution entry. Its batched gather implementation is
    separate from the pointwise CPU golden; only container and contract helpers
    are shared. Device precision still needs runtime comparison.

    Uses ``torch.floor`` and per-corner zero padding with fp32 accumulation,
    then restores the requested output dtype. Only modulated=True is supported.
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
        device=None,
        **kwargs,
    ):
        self.strides, self.pads, self.ksize = tuple(strides), tuple(pads), tuple(ksize)
        self.dilations, self.data_format = tuple(dilations), data_format
        self.modulated = modulated
        self.device = device
        self.output_dtypes = kwargs.get("output_dtypes")
        self.stride_h, self.stride_w = _hw_components(strides, data_format)
        self.dil_h, self.dil_w = _hw_components(dilations, data_format)
        self.pad_u, self.pad_l = int(pads[0]), int(pads[2])
        self.k_h, self.k_w = int(ksize[0]), int(ksize[1])
        self.groups = int(deformable_groups)

    def __call__(self, x, offsets, **kwargs):
        x_tensor = _as_tensor(x, device=self.device)
        out_dtype = _resolve_output_dtype(
            x_tensor.dtype,
            output_dtypes=kwargs.get("output_dtypes", self.output_dtypes),
        )
        x = x_tensor.to(torch.float32)
        offsets = _as_tensor(offsets, device=x.device).to(torch.float32)
        _validate_inputs(
            x,
            offsets,
            self.strides,
            self.pads,
            self.ksize,
            self.dilations,
            self.data_format,
            self.groups,
            self.modulated,
        )
        n, in_h, in_w, in_c = x.shape
        _, out_h, out_w, _ = offsets.shape

        k_h, k_w = self.k_h, self.k_w
        cpg = in_c // self.groups
        comp_stride = self.groups * k_h * k_w
        group_stride = k_h * k_w

        y = x.new_zeros((n, out_h * k_h, out_w * k_w, in_c))
        if in_h == 0 or in_w == 0:
            # There are no valid neighbours in an empty input image. Validation
            # above still checks structure and rejects non-finite offsets/masks.
            return [y.to(dtype=_torch_dtype(out_dtype))]

        oh = torch.arange(out_h, device=x.device, dtype=torch.int64).view(
            1, out_h, 1, 1, 1
        )
        ow = torch.arange(out_w, device=x.device, dtype=torch.int64).view(
            1, 1, out_w, 1, 1
        )
        kh = torch.arange(k_h, device=x.device, dtype=torch.int64).view(1, 1, 1, k_h, 1)
        kw = torch.arange(k_w, device=x.device, dtype=torch.int64).view(1, 1, 1, 1, k_w)
        bn = torch.arange(n, device=x.device, dtype=torch.int64).view(n, 1, 1, 1, 1)

        for g in range(self.groups):
            bx = 0 * comp_stride + g * group_stride
            by = 1 * comp_stride + g * group_stride
            bm = 2 * comp_stride + g * group_stride
            off_x = offsets[:, :, :, bx : bx + group_stride].reshape(
                n, out_h, out_w, k_h, k_w
            )
            off_y = offsets[:, :, :, by : by + group_stride].reshape(
                n, out_h, out_w, k_h, k_w
            )
            mask = offsets[:, :, :, bm : bm + group_stride].reshape(
                n, out_h, out_w, k_h, k_w
            )

            point_h = (oh * self.stride_h - self.pad_u).to(torch.float32)
            point_h = point_h + (kh * self.dil_h).to(torch.float32) + off_y
            point_w = (ow * self.stride_w - self.pad_l).to(torch.float32)
            point_w = point_w + (kw * self.dil_w).to(torch.float32) + off_x

            h0 = torch.floor(point_h)
            w0 = torch.floor(point_w)
            dh = (point_h - h0).unsqueeze(-1)
            dw = (point_w - w0).unsqueeze(-1)

            xg = x[:, :, :, g * cpg : (g + 1) * cpg]

            acc = None
            for dhh, wh in ((0, 1.0 - dh), (1, dh)):
                hh = h0 + dhh
                for dww, ww_ in ((0, 1.0 - dw), (1, dw)):
                    ww = w0 + dww
                    valid_h = (hh >= 0) & (hh < in_h)
                    valid_w = (ww >= 0) & (ww < in_w)
                    valid = valid_h & valid_w
                    # Mask in floating point BEFORE integer conversion: casting
                    # a finite coordinate such as 3e38 to int64 can overflow.
                    hhc = torch.where(valid_h, hh, torch.zeros_like(hh)).to(torch.int64)
                    wwc = torch.where(valid_w, ww, torch.zeros_like(ww)).to(torch.int64)
                    v = xg[bn, hhc, wwc, :]
                    v = torch.where(valid.unsqueeze(-1), v, torch.zeros_like(v))
                    term = (wh * ww_) * v
                    acc = term if acc is None else acc + term

            acc = acc * mask.unsqueeze(-1)

            out_row = oh * k_h + kh
            out_col = ow * k_w + kw
            y[bn, out_row, out_col, g * cpg : (g + 1) * cpg] = acc

        return [y.to(dtype=_torch_dtype(out_dtype))]


class DeformableOffsetsKernelSpec:
    """Kernel / GEIR spec — pointwise Torch golden and batched Torch reference."""

    golden = staticmethod(deformable_offsets_golden)
    third_party = {"torch": DeformableOffsetsTorchCompetitor}
    customize_inputs = staticmethod(deformable_offsets_input)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
