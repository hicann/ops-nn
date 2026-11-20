#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK e2e (torch) plugin for torch.ops.npu.npu_deformable_conv2d.

Provides:
  - golden: modulated deformable conv (DCNv2) CPU reference. Offsets use the
    value-block layout (dx block | dy block | mask block, each dfm*kh*kw channels),
    matching the DeformableOffsets kernel semantics; out-of-bound samples are 0.
  - torchair fx2ge converter: maps npu_deformable_conv2d to the graph-interface
    DeformableConv2D op (plus DeformableOffsets for the deform_out output), so
    GE graph legs exercise the migrated infershape + ADeformableConv2dPass
    decomposition instead of graph-breaking to the aclnn kernel path.
"""

import torch
import torch.nn.functional as F

from torch_npu.dynamo.torchair._ge_concrete_graph.fx2ge_converter import (
    register_fx_node_ge_converter,
)
from torch_npu.dynamo.torchair._ge_concrete_graph import ge_apis as ge


def _bilinear_sample(x, ph, pw):
    """Bilinear sample of x [n,c,ih,iw] at fractional positions ph/pw [n,oh,ow].

    Positions are in ORIGINAL (unpadded) x coordinates; the convolution padding
    is zero-padding, so each of the four interpolation neighbors contributes
    x only when its index is inside [0, ih-1] x [0, iw-1], else 0.
    """
    n, c, ih, iw = x.shape
    oh, ow = ph.shape[1], pw.shape[2]
    h0 = torch.floor(ph).long()
    w0 = torch.floor(pw).long()
    vh = (ph - h0).view(n, 1, oh, ow)
    vw = (pw - w0).view(n, 1, oh, ow)
    sample = torch.zeros(n, c, oh, ow)
    for ha, wa, fa, fb in (
        (h0, w0, 1 - vh, 1 - vw),
        (h0, w0 + 1, 1 - vh, vw),
        (h0 + 1, w0, vh, 1 - vw),
        (h0 + 1, w0 + 1, vh, vw),
    ):
        valid = (ha >= 0) & (ha <= ih - 1) & (wa >= 0) & (wa <= iw - 1)
        if not valid.any():
            continue
        gidx = (ha.clamp(0, ih - 1) * iw + wa.clamp(0, iw - 1)).view(n, 1, oh * ow)
        gathered = (
            x.reshape(n, c, ih * iw)
            .gather(2, gidx.expand(n, c, oh * ow))
            .view(n, c, oh, ow)
        )
        sample = sample + torch.where(
            valid.view(n, 1, oh, ow), gathered * fa * fb, torch.zeros(())
        )
    return sample


class NpuDeformableConv2dTorchSpec:
    """torch.ops.npu.npu_deformable_conv2d e2e spec: DCNv2 golden + GE converter."""

    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }

    def golden(
        input,
        weight,
        offset,
        bias,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        deformable_groups,
        modulated,
        **kwargs,
    ):
        x = input.float()
        w = weight.float()
        off = offset.float()
        b = bias.float() if bias is not None else None
        n, c, h, wid = x.shape
        kn, kc, kh, kw = w.shape
        sh, sw = stride[2], stride[3]
        pt, pb = padding[0], padding[1]
        pl, pr = padding[2], padding[3]
        dh, dw = dilation[2], dilation[3]
        oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
        ow = (wid + pl + pr - dw * (kw - 1) - 1) // sw + 1
        kk = kh * kw
        dfm = deformable_groups
        dx = off[:, 0 * dfm * kk : 1 * dfm * kk, :, :]
        dy = off[:, 1 * dfm * kk : 2 * dfm * kk, :, :]
        mk = off[:, 2 * dfm * kk : 3 * dfm * kk, :, :] if modulated else None
        cg = c // dfm
        base_h = torch.arange(oh).view(oh, 1).expand(oh, ow) * sh - pt
        base_w = torch.arange(ow).view(1, ow).expand(oh, ow) * sw - pl
        # per kernel point (g, i, j): sampled feature map [n, cg, oh, ow]
        planes = [[[None] * kw for _ in range(kh)] for _ in range(dfm)]
        for g in range(dfm):
            for i in range(kh):
                for j in range(kw):
                    kidx = g * kk + i * kw + j
                    ph = base_h + i * dh + dy[:, kidx]
                    pw = base_w + j * dw + dx[:, kidx]
                    sample = _bilinear_sample(x[:, g * cg : (g + 1) * cg], ph, pw)
                    if mk is not None:
                        sample = sample * mk[:, kidx].view(n, 1, oh, ow)
                    planes[g][i][j] = sample
        # assemble deform [n, c, oh*kh, ow*kw] via [n, dfm, cg, kh, oh, kw, ow]
        deform5 = torch.zeros(n, dfm, cg, kh, oh, kw, ow)
        for g in range(dfm):
            for i in range(kh):
                for j in range(kw):
                    deform5[:, g, :, i, :, j, :] = planes[g][i][j]
        deform = deform5.permute(0, 1, 2, 4, 3, 6, 5).reshape(n, c, oh * kh, ow * kw)
        y = F.conv2d(
            deform, w, b, stride=(kh, kw), padding=0, dilation=1, groups=groups
        )
        return [y, deform]


if not hasattr(torch.ops.npu.npu_deformable_conv2d.default, "_ge_converter"):

    @register_fx_node_ge_converter(torch.ops.npu.npu_deformable_conv2d.default)
    def _converter_npu_deformable_conv2d(
        input,
        weight,
        offset,
        bias,
        kernel_size,
        stride,
        padding,
        dilation=(1, 1, 1, 1),
        groups=1,
        deformable_groups=1,
        modulated=True,
        meta_outputs=None,
    ):
        deform = ge.DeformableOffsets(
            input,
            offsets=offset,
            strides=stride,
            pads=padding,
            ksize=kernel_size,
            dilations=dilation,
            data_format="NCHW",
            deformable_groups=deformable_groups,
            modulated=modulated,
        )
        y = ge.DeformableConv2D(
            input,
            filter=weight,
            offsets=offset,
            bias=bias,
            strides=stride,
            pads=padding,
            dilations=dilation,
            groups=groups,
            data_format="NCHW",
            deformable_groups=deformable_groups,
            modulated=modulated,
        )
        # fx2ge fills output descs from fake-tensor meta (dtype/shape only, no format).
        # The pass copies the y desc into the DeformableOffsets output, and Conv2DV2
        # (the decomposed child) has NCHW/NHWC-only kernels, so the output formats
        # must be pinned here or the Conv2DV2 tiling key selects a format-less kernel.
        for out in (y, deform):
            out.desc.attr["format_for_int"].i = 0  # FORMAT_NCHW
            out.desc.attr["origin_format_for_int"].i = 0  # FORMAT_NCHW
            out.desc.attr["origin_format_is_set"].b = True
        return (y, deform)


__spec__ = {"torch.ops.npu.npu_deformable_conv2d": "NpuDeformableConv2dTorchSpec"}
