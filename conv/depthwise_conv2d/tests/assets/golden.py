#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK golden plugin for DepthwiseConv2D (kernel/geir flow).

golden 数学：深度卷积 = groups=C_in 的分组卷积，参考实现用 torch.nn.functional.conv2d
在 CPU 上计算，fp16/bf16 输入先升 fp32 计算再转回，保证 golden 精度。
"""

__spec__ = {"depthwise_conv2d": "DepthwiseConv2DTestSpec"}

import numpy as np
import torch
import torch.nn.functional as F


class DepthwiseConv2DTestSpec:
    @staticmethod
    def golden(x, w, *args, **kwargs):
        strides = list(kwargs.get("strides", [1, 1, 1, 1]))
        pads = list(kwargs.get("pads", [0, 0, 0, 0]))
        dilations = list(kwargs.get("dilations", [1, 1, 1, 1]))
        data_format = kwargs.get("data_format", "NCHW")
        padding_mode = str(kwargs.get("padding", "") or "").upper()
        bias = args[0] if args and args[0] is not None else None

        # 量化输入输出 INT32；fp16/bf16 升 fp32 比对（容差覆盖低精度舍入）；其余与输入一致
        if x.dtype in (np.int8, np.int16):
            out_dtype = np.int32
        elif x.dtype in (np.float16,) or str(x.dtype) == "bfloat16":
            out_dtype = np.float32
        else:
            out_dtype = x.dtype

        # 空/退化输入防线：TTK 版本差异可能以非 4D 形态传空 filter（隔离环境实测崩溃），
        # 统一返回空数组（0 元素），与空输出图（0 字节）按 size=0 对齐
        if (
            x is None
            or w is None
            or x.ndim < 2
            or w.ndim < 2
            or x.size == 0
            or w.size == 0
        ):
            return [np.zeros((0,), dtype=out_dtype)]
        xt = torch.from_numpy(np.ascontiguousarray(x)).float()
        wt = torch.from_numpy(np.ascontiguousarray(w)).float()
        bt = (
            torch.from_numpy(np.ascontiguousarray(bias)).float()
            if bias is not None
            else None
        )

        if data_format == "NHWC":
            # x: [N,H,W,C] -> [N,C,H,W]
            xt = xt.permute(0, 3, 1, 2)
            # filter HWCN [kh,kw,C,M] -> [C*M,1,kh,kw]（组主序）
            kh, kw, _, _ = wt.shape
            wt = wt.permute(2, 3, 0, 1).reshape(-1, 1, kh, kw)
            sh, sw = strides[1], strides[2]
            dh, dw = dilations[1], dilations[2]
        else:
            # filter NCHW [C*M,1,kh,kw] 直接可用
            sh, sw = strides[2], strides[3]
            dh, dw = dilations[2], dilations[3]
        groups = xt.shape[1]
        in_h, in_w = int(xt.shape[2]), int(xt.shape[3])
        kh, kw = int(wt.shape[2]), int(wt.shape[3])

        # padding attr 覆盖 pads（与 infershape 三层取值一致：_padding > "padding" > pads）
        if padding_mode == "SAME":
            pt, pb, pl, pr = DepthwiseConv2DTestSpec._same_pads(
                in_h, in_w, kh, kw, sh, sw, dh, dw
            )
        elif padding_mode == "VALID":
            pt, pb, pl, pr = 0, 0, 0, 0
        else:
            pt, pb, pl, pr = pads[0], pads[1], pads[2], pads[3]

        if pt != pb or pl != pr:
            xt = F.pad(xt, (int(pl), int(pr), int(pt), int(pb)))
            pad2 = (0, 0)
        else:
            pad2 = (int(pt), int(pl))

        # 零张量/零输出通道守卫：depthwise outC=filterN*filterC，为 0 时直接给零数组
        if data_format == "NHWC":
            out_c = int(w.shape[2]) * int(w.shape[3])
        else:
            out_c = int(w.shape[0]) * int(w.shape[1])
        if out_c == 0 or in_h == 0 or in_w == 0:
            oh = max((in_h + pt + pb - dh * (kh - 1) - 1) // sh + 1, 0)
            ow = max((in_w + pl + pr - dw * (kw - 1) - 1) // sw + 1, 0)
            if data_format == "NHWC":
                shape = (xt.shape[0], oh, ow, out_c)
            else:
                shape = (xt.shape[0], out_c, oh, ow)
            return [np.zeros(shape).astype(out_dtype)]

        yt = F.conv2d(
            xt,
            wt,
            bt,
            stride=(int(sh), int(sw)),
            padding=pad2,
            dilation=(int(dh), int(dw)),
            groups=int(groups),
        )
        if data_format == "NHWC":
            yt = yt.permute(0, 2, 3, 1)
        return [yt.numpy().astype(out_dtype)]

    @staticmethod
    def _same_pads(in_h, in_w, kh, kw, sh, sw, dh, dw):
        def one(i, k, s, d):
            eff = (k - 1) * d + 1
            out = -(-i // s)
            total = max((out - 1) * s + eff - i, 0)
            before = total // 2
            return before, total - before

        pt, pb = one(in_h, kh, sh, dh)
        pl, pr = one(in_w, kw, sw, dw)
        return pt, pb, pl, pr
