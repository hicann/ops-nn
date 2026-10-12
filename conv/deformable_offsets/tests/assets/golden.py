#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden for the DeformableOffsets forward operator (TTK kernel path).

The reference reproduces the arch35 kernel semantics exactly:
  * NHWC tensors, offsets channel layout ``[component(0=x, 1=y, 2=mask), group, kh, kw]``;
  * sampled coordinate ``pointH = oh*strideH - padU + kh*dilH + offY`` (W analogous);
  * bilinear interpolation with per-neighbour out-of-bound -> 0;
  * ``y[n, oh*kH+kh, ow*kW+kw, c] = mask * sum(weight_ij * x[n, hi, wj, c])``.
Low precision (fp16/bf16) is computed in fp32 and cast back to the output dtype,
mirroring the kernel float accumulation.
"""

import numpy as np


__input__ = {"kernel": {"deformable_offsets": "deformable_offsets_input"}}
__golden__ = {"kernel": {"deformable_offsets": "deformable_offsets_golden"}}


def _is_bf16(arr):
    return arr.dtype.name == "bfloat16"


def _to_fp32(arr):
    """Convert an input numpy array (fp16/bf16/fp32) to fp32 for computation."""
    if _is_bf16(arr):
        import ml_dtypes  # noqa: F401  (registers the bfloat16 numpy dtype)

        return arr.astype(np.float32)
    return arr.astype(np.float32)


def _cast_back(arr_fp32, dst_dtype):
    """Cast an fp32 result back to the requested output dtype."""
    if dst_dtype is None:
        return arr_fp32
    if isinstance(dst_dtype, str):
        name = dst_dtype
    else:
        name = np.dtype(dst_dtype).name
    if name == "bfloat16":
        import ml_dtypes

        return arr_fp32.astype(ml_dtypes.bfloat16)
    return arr_fp32.astype(np.dtype(name))


def _hw_components(seq, data_format):
    """Extract the (H, W) entries from a 4-length NHWC/NCHW attribute list."""
    seq = list(seq)
    h_idx = data_format.index("H")
    w_idx = data_format.index("W")
    return seq[h_idx], seq[w_idx]


def _resolve_output_dtype(default_dtype, **kwargs):
    out_dtypes = kwargs.get("output_dtypes")
    if out_dtypes:
        return out_dtypes[0]
    return default_dtype


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
    out_dtype = _resolve_output_dtype(x.dtype, **kwargs)

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

    # Shape self-consistency (matches infershape contract).
    exp_out_h = (in_h + pad_u + pad_d - ((k_h - 1) * dil_h + 1)) // stride_h + 1
    exp_out_w = (in_w + pad_l + pad_r - ((k_w - 1) * dil_w + 1)) // stride_w + 1
    assert (exp_out_h, exp_out_w) == (out_h, out_w), (
        f"offsets H/W {(out_h, out_w)} != conv out {(exp_out_h, exp_out_w)}"
    )
    assert off_c == 3 * groups * k_h * k_w, (
        f"offsets channel {off_c} != 3*groups*kH*kW={3 * groups * k_h * k_w}"
    )
    assert in_c % groups == 0, f"in_c {in_c} not divisible by groups {groups}"
    cpg = in_c // groups  # channels per group

    comp_stride = groups * k_h * k_w  # stride between (x, y, mask) components
    group_stride = k_h * k_w

    y = np.zeros((n, out_h * k_h, out_w * k_w, in_c), dtype=np.float32)

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

                            point_h = oh * stride_h - pad_u + kh * dil_h + off_y
                            point_w = ow * stride_w - pad_l + kw * dil_w + off_x

                            h0 = int(np.floor(point_h))
                            w0 = int(np.floor(point_w))
                            dh = point_h - h0
                            dw = point_w - w0

                            gc = slice(g * cpg, (g + 1) * cpg)
                            acc = np.zeros(cpg, dtype=np.float32)
                            # four neighbours with per-corner boundary -> 0
                            for hh, wgt_h in ((h0, 1.0 - dh), (h0 + 1, dh)):
                                if hh < 0 or hh >= in_h:
                                    continue
                                for ww, wgt_w in ((w0, 1.0 - dw), (w0 + 1, dw)):
                                    if ww < 0 or ww >= in_w:
                                        continue
                                    acc += (wgt_h * wgt_w) * x_f[bn, hh, ww, gc]

                            out_row = oh * k_h + kh
                            out_col = ow * k_w + kw
                            y[bn, out_row, out_col, gc] = mask * acc

    return [_cast_back(y, out_dtype)]


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
