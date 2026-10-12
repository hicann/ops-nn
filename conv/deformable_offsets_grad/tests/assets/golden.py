#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden for the DeformableOffsetsGrad backward operator (TTK kernel path).

The reference reproduces the arch35 grad kernel exactly (explicit analytic
derivative of the forward bilinear sampling, NOT autodiff):

  For each (n, oh, ow, group, kh, kw):
    originH = oh*strideH - padU + kh*dilH + offY   (W analogous)
    only if -1 < originH < inH and -1 < originW < inW, else grad_offsets = 0.
    h0=floor(originH), w0=floor(originW); lh=originH-h0, lw=originW-w0;
    hh=1-lh, hw=1-lw; weights w1=hh*hw, w2=hh*lw, w3=lh*hw, w4=lh*lw at
    (h0,w0),(h0,w0+1),(h0+1,w0),(h0+1,w0+1) with per-corner boundary guards.

    grad_x[corner, c] += grad[out_pixel, c] * mask * w_corner            (scatter add)
    gH   = sum_c grad[c]*mask*(-hw*v1 - lw*v2 + hw*v3 + lw*v4)
    gW   = sum_c grad[c]*mask*(-hh*v1 + hh*v2 - lh*v3 + lh*v4)
    gMask= sum_c grad[c]*(w1*v1 + w2*v2 + w3*v3 + w4*v4)

grad_offsets keeps the same channel layout as offsets
(``[component(0=x->gW, 1=y->gH, 2=mask->gMask), group, kh, kw]``).
Low precision is computed in fp32 and cast back, mirroring the kernel.
"""

import numpy as np


__input__ = {"kernel": {"deformable_offsets_grad": "deformable_offsets_grad_input"}}
__golden__ = {"kernel": {"deformable_offsets_grad": "deformable_offsets_grad_golden"}}


def _is_bf16(arr):
    return arr.dtype.name == "bfloat16"


def _to_fp32(arr):
    if _is_bf16(arr):
        import ml_dtypes  # noqa: F401

        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _cast_back(arr_fp32, dst_dtype):
    if dst_dtype is None:
        return arr_fp32
    name = dst_dtype if isinstance(dst_dtype, str) else np.dtype(dst_dtype).name
    if name == "bfloat16":
        import ml_dtypes

        return arr_fp32.astype(ml_dtypes.bfloat16)
    return arr_fp32.astype(np.dtype(name))


def _hw_components(seq, data_format):
    seq = list(seq)
    return seq[data_format.index("H")], seq[data_format.index("W")]


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
    """Kernel golden for ``deformable_offsets_grad``.

    Parameter names/order follow ``deformable_offsets_grad_def.cpp``
    (inputs grad, x, offsets then attrs). Inputs are ``numpy.ndarray``.
    Returns ``[grad_x, grad_offsets]``.
    """
    out_dtypes = kwargs.get("output_dtypes")
    gx_dtype = out_dtypes[0] if out_dtypes else x.dtype
    goff_dtype = (
        out_dtypes[1] if (out_dtypes and len(out_dtypes) > 1) else offsets.dtype
    )

    grad_f = _to_fp32(grad)
    x_f = _to_fp32(x)
    off_f = _to_fp32(offsets)

    n, in_h, in_w, in_c = x_f.shape
    _, out_h, out_w, off_c = off_f.shape

    k_h, k_w = int(ksize[0]), int(ksize[1])
    stride_h, stride_w = _hw_components(strides, data_format)
    dil_h, dil_w = _hw_components(dilations, data_format)
    pad_u, pad_d, pad_l, pad_r = (int(p) for p in pads)
    stride_h, stride_w = int(stride_h), int(stride_w)
    dil_h, dil_w = int(dil_h), int(dil_w)
    groups = int(deformable_groups)

    assert off_c == 3 * groups * k_h * k_w, "unexpected offsets channel count"
    assert in_c % groups == 0, "in_c not divisible by groups"
    cpg = in_c // groups

    comp_stride = groups * k_h * k_w
    group_stride = k_h * k_w

    grad_x = np.zeros((n, in_h, in_w, in_c), dtype=np.float32)
    grad_off = np.zeros((n, out_h, out_w, off_c), dtype=np.float32)

    for bn in range(n):
        for oh in range(out_h):
            for ow in range(out_w):
                for kh in range(k_h):
                    for kw in range(k_w):
                        base = kh * k_w + kw
                        out_row = oh * k_h + kh
                        out_col = ow * k_w + kw
                        for g in range(groups):
                            ch = g * group_stride + base
                            x_ch = 0 * comp_stride + ch
                            y_ch = 1 * comp_stride + ch
                            m_ch = 2 * comp_stride + ch

                            off_x = off_f[bn, oh, ow, x_ch]
                            off_y = off_f[bn, oh, ow, y_ch]
                            mask = off_f[bn, oh, ow, m_ch]

                            origin_h = oh * stride_h - pad_u + kh * dil_h + off_y
                            origin_w = ow * stride_w - pad_l + kw * dil_w + off_x

                            # outer differentiability gate; else grad_offsets stays 0
                            if not (-1 < origin_h < in_h and -1 < origin_w < in_w):
                                continue

                            h0 = int(np.floor(origin_h))
                            w0 = int(np.floor(origin_w))
                            lh = origin_h - h0
                            lw = origin_w - w0
                            hh = 1.0 - lh
                            hw = 1.0 - lw

                            gc = slice(g * cpg, (g + 1) * cpg)
                            g_vec = grad_f[bn, out_row, out_col, gc]  # (cpg,)

                            g_h = 0.0
                            g_w = 0.0
                            g_mask = 0.0

                            # corner1 (h0, w0): weight hh*hw
                            if h0 >= 0 and w0 >= 0:
                                v = x_f[bn, h0, w0, gc]
                                w1 = hh * hw
                                grad_x[bn, h0, w0, gc] += g_vec * (mask * w1)
                                g_h += float(np.sum(g_vec * (-v * hw)))
                                g_w += float(np.sum(g_vec * (-v * hh)))
                                g_mask += float(np.sum(g_vec * (w1 * v)))
                            # corner2 (h0, w0+1): weight hh*lw
                            if h0 >= 0 and (w0 + 1) < in_w:
                                v = x_f[bn, h0, w0 + 1, gc]
                                w2 = hh * lw
                                grad_x[bn, h0, w0 + 1, gc] += g_vec * (mask * w2)
                                g_h += float(np.sum(g_vec * (-v * lw)))
                                g_w += float(np.sum(g_vec * (v * hh)))
                                g_mask += float(np.sum(g_vec * (w2 * v)))
                            # corner3 (h0+1, w0): weight lh*hw
                            if (h0 + 1) < in_h and w0 >= 0:
                                v = x_f[bn, h0 + 1, w0, gc]
                                w3 = lh * hw
                                grad_x[bn, h0 + 1, w0, gc] += g_vec * (mask * w3)
                                g_h += float(np.sum(g_vec * (v * hw)))
                                g_w += float(np.sum(g_vec * (-v * lh)))
                                g_mask += float(np.sum(g_vec * (w3 * v)))
                            # corner4 (h0+1, w0+1): weight lh*lw
                            if (h0 + 1) < in_h and (w0 + 1) < in_w:
                                v = x_f[bn, h0 + 1, w0 + 1, gc]
                                w4 = lh * lw
                                grad_x[bn, h0 + 1, w0 + 1, gc] += g_vec * (mask * w4)
                                g_h += float(np.sum(g_vec * (v * lw)))
                                g_w += float(np.sum(g_vec * (v * lh)))
                                g_mask += float(np.sum(g_vec * (w4 * v)))

                            grad_off[bn, oh, ow, y_ch] = mask * g_h
                            grad_off[bn, oh, ow, x_ch] = mask * g_w
                            grad_off[bn, oh, ow, m_ch] = g_mask

    return [_cast_back(grad_x, gx_dtype), _cast_back(grad_off, goff_dtype)]


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
    """Custom input generator for special DeformableOffsetsGrad cases.

    Keyed on ``testcase_name``:
      * ``*interior*`` -> set x/y offsets to small fractional values and mask to 1
        so sampled points stay strictly interior and non-integer
        (finite-difference friendly, numerically stable on device).
    Otherwise inputs are returned unchanged.
    """
    name = kwargs.get("testcase_name", "") or ""
    if "interior" not in name:
        return [grad, x, offsets]

    k_h, k_w = int(ksize[0]), int(ksize[1])
    groups = int(deformable_groups)
    comp_stride = groups * k_h * k_w

    new_off = np.array(offsets, copy=True)
    _, out_h, out_w, off_c = new_off.shape
    # small, deterministic-yet-varied fractional offsets in [0.25, 0.45]
    per = comp_stride
    ramp = (0.25 + 0.20 * (np.arange(per) / max(per - 1, 1))).astype(np.float32)
    for oh in range(out_h):
        for ow in range(out_w):
            new_off[:, oh, ow, 0 * comp_stride : 1 * comp_stride] = ramp  # x offset
            new_off[:, oh, ow, 1 * comp_stride : 2 * comp_stride] = ramp  # y offset
    one = np.ones(1, dtype=new_off.dtype)[0]
    new_off[..., 2 * comp_stride : 3 * comp_stride] = one  # mask = 1
    return [grad, x, new_off.astype(offsets.dtype)]
