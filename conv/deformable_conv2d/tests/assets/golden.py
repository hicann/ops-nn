#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden plugin for DeformableConv2D (kernel/geir flow).

参考实现按算子分解语义计算：DeformableOffsets + Conv2D(stride = kernel_size)。

offsets 通道为 value-block 布局 ``[dx block | dy block | mask block]``，每个 block
``deformable_groups * kh * kw`` 个通道（组主序，组内 kernel 行主序）；``modulated``
为 False 时无 mask block。卷积 padding 为零填充：双线性插值的四个邻域只在原始图像
``[0, ih-1] x [0, iw-1]`` 内取值，越界邻域贡献 0。
"""

__spec__ = {"deformable_conv2_d": "DeformableConv2DTestSpec"}

import numpy as np


def _bilinear_sample(x, ph, pw):
    """按分数坐标采样 x ``[n, c, ih, iw]``。

    ``ph`` / ``pw`` 为原始（未 pad）坐标系下的采样位置，形状 ``[n, oh, ow]``；
    邻域越出 ``[0, ih-1] x [0, iw-1]`` 时取 0，与零填充卷积语义一致。
    """
    n, c, ih, iw = x.shape
    oh, ow = ph.shape[1], pw.shape[2]
    h0 = np.floor(ph).astype(np.int64)
    w0 = np.floor(pw).astype(np.int64)
    vh = (ph - h0)[:, None, :, :]
    vw = (pw - w0)[:, None, :, :]
    sample = np.zeros((n, c, oh, ow), dtype=np.float64)
    flat = x.reshape(n, c, ih * iw)
    for ha, wa, fa, fb in (
        (h0, w0, 1 - vh, 1 - vw),
        (h0, w0 + 1, 1 - vh, vw),
        (h0 + 1, w0, vh, 1 - vw),
        (h0 + 1, w0 + 1, vh, vw),
    ):
        valid = (ha >= 0) & (ha <= ih - 1) & (wa >= 0) & (wa <= iw - 1)
        if not valid.any():
            continue
        gidx = (np.clip(ha, 0, ih - 1) * iw + np.clip(wa, 0, iw - 1)).reshape(
            n, 1, oh * ow
        )
        gathered = np.take_along_axis(
            flat, np.broadcast_to(gidx, (n, c, oh * ow)), axis=2
        ).reshape(n, c, oh, ow)
        sample += np.where(valid[:, None, :, :], gathered * fa * fb, 0.0)
    return sample


def _output_dtype(x):
    """fp16/bf16 输入以 fp32 输出参与比对（容差覆盖低精度舍入），其余与输入一致。"""
    if x.dtype in (np.float16,):
        return np.float32
    if str(x.dtype) == "bfloat16":
        return np.float32
    if x.dtype in (np.int8, np.int16):
        return np.int32
    return x.dtype


class DeformableConv2DTestSpec:
    @staticmethod
    def golden(x, w, *args, **kwargs):
        offsets = args[0] if len(args) > 0 else None
        bias = args[1] if len(args) > 1 else None

        attrs = kwargs.get("attributes", kwargs) or {}
        strides = list(attrs.get("strides", [1, 1, 1, 1]))
        pads = list(attrs.get("pads", [0, 0, 0, 0]))
        dilations = list(attrs.get("dilations", [1, 1, 1, 1]))
        groups = int(attrs.get("groups", 1) or 1)
        dfm = int(attrs.get("deformable_groups", 1) or 1)
        modulated = bool(attrs.get("modulated", True))
        data_format = str(attrs.get("data_format", "NCHW") or "NCHW").upper()

        out_dtype = _output_dtype(x)
        if x is None or w is None or x.ndim < 4 or w.ndim < 4:
            return [np.zeros((0,), dtype=out_dtype)]

        x = np.ascontiguousarray(x).astype(np.float64)
        w = np.ascontiguousarray(w).astype(np.float64)
        off = None
        if offsets is not None:
            off = np.ascontiguousarray(offsets).astype(np.float64)

        nhwc = DeformableConv2DTestSpec._resolve_nhwc(
            x, w, off, data_format, dfm, modulated
        )
        if nhwc:
            x = np.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
            if off is not None:
                off = np.transpose(off, (0, 3, 1, 2))  # NHWC -> NCHW
            w = np.transpose(w, (3, 2, 0, 1))  # HWCN -> NCHW
            sh, sw = int(strides[1]), int(strides[2])
            dh, dw = int(dilations[1]), int(dilations[2])
        else:
            sh, sw = int(strides[2]), int(strides[3])
            dh, dw = int(dilations[2]), int(dilations[3])
        pt, pb, pl, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])

        n, ic, ih, iw = x.shape
        kn, kc, kh, kw = w.shape
        oh = (ih + pt + pb - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + pl + pr - dw * (kw - 1) - 1) // sw + 1
        oh, ow = max(oh, 0), max(ow, 0)

        def _empty_like():
            shape = (n, oh, ow, kn) if nhwc else (n, kn, oh, ow)
            return [np.zeros(shape, dtype=out_dtype)]

        if (
            oh == 0
            or ow == 0
            or kn == 0
            or kh == 0
            or kw == 0
            or ic == 0
            or x.size == 0
        ):
            return _empty_like()

        kk = kh * kw
        if off is None:
            off = np.zeros(
                (n, (3 if modulated else 2) * dfm * kk, oh, ow), dtype=np.float64
            )
            if modulated:
                off[:, 2 * dfm * kk :, :, :] = 1.0
        elif off.shape[1] == 2 * dfm * kk:
            modulated = False  # 2-block offsets 即 DCNv1，与 attr 缺省无关

        dx = off[:, 0 : dfm * kk, :, :]
        dy = off[:, dfm * kk : 2 * dfm * kk, :, :]
        mk = off[:, 2 * dfm * kk : 3 * dfm * kk, :, :] if modulated else None

        base_h = np.arange(oh).reshape(oh, 1) * sh - pt
        base_w = np.arange(ow).reshape(1, ow) * sw - pl
        cg = ic // dfm
        ocg = kn // groups
        deformed = np.zeros((n, dfm, cg, kh, oh, kw, ow), dtype=np.float64)
        for g in range(dfm):
            for i in range(kh):
                for j in range(kw):
                    kidx = g * kk + i * kw + j
                    ph = np.broadcast_to(base_h, (n, oh, ow)) + i * dh + dy[:, kidx]
                    pw = np.broadcast_to(base_w, (n, oh, ow)) + j * dw + dx[:, kidx]
                    sample = _bilinear_sample(x[:, g * cg : (g + 1) * cg], ph, pw)
                    if mk is not None:
                        sample = sample * mk[:, kidx][:, None, :, :]
                    deformed[:, g, :, i, :, j, :] = sample
        # [n, dfm, cg, kh, oh, kw, ow] -> [n, ic, oh*kh, ow*kw]
        deform = deformed.transpose(0, 1, 2, 4, 3, 6, 5).reshape(
            n, ic, oh * kh, ow * kw
        )
        blocks = deform.reshape(n, ic, oh, kh, ow, kw)

        y = np.zeros((n, kn, oh, ow), dtype=np.float64)
        for og in range(groups):
            for oci in range(ocg):
                oc = og * ocg + oci
                acc = np.zeros((n, oh, ow), dtype=np.float64)
                for ci in range(kc):
                    icc = og * kc + ci
                    for i in range(kh):
                        for j in range(kw):
                            acc += blocks[:, icc, :, i, :, j] * w[oc, ci, i, j]
                y[:, oc, :, :] = acc
        if bias is not None:
            y += np.ascontiguousarray(bias).astype(np.float64).reshape(1, kn, 1, 1)

        if nhwc:
            y = np.transpose(y, (0, 2, 3, 1))  # NCHW -> NHWC
        return [y.astype(out_dtype)]

    @staticmethod
    def _resolve_nhwc(x, w, off, data_format, dfm, modulated):
        """判定输入实际排布。

        ``data_format`` 属性与实际张量可能冲突（如属性写 NHWC 而张量为 NCHW），
        此时按张量实际排布计算，与 infershape 以输入 origin format 优先的规则一致。
        """
        if data_format != "NHWC" or x.ndim != 4 or w.ndim != 4:
            return False
        kk = int(w.shape[0]) * int(w.shape[1])  # HWCN: 前两维是 kernel
        exp_off_c = (3 if modulated else 2) * dfm * kk
        c_ok = x.shape[3] == int(w.shape[2])  # x.C == filter.C
        off_ok = off is None or (off.ndim == 4 and off.shape[3] == exp_off_c)
        return bool(c_ok and off_ok)
