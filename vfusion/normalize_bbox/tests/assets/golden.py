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

import numpy as np
import torch

__golden__ = {
    "kernel": {"normalize_bbox": "normalize_bbox_golden"},
}


def _compute(boxes, shape_hw, reversed_box):
    """Core torch implementation, mirrors the kernel cast chain.

    y = boxes / [h, w, h, w]  (per batch, h=shapeHw[b,0], w=shapeHw[b,1], int32->float)

    The kernel computes in two dtype-dependent paths; the golden must
    replicate each cast chain to avoid half-ulp systematic bias:

      - fp16 : h/w int32->f32->half, divisor half, Div on half        (CompT == half)
      - fp32 : h/w int32->f32,       divisor f32,  Div on float        (CompT == float)
    """
    if not hasattr(boxes, "detach"):
        boxes = torch.from_numpy(np.ascontiguousarray(boxes))
    if not hasattr(shape_hw, "detach"):
        shape_hw = torch.from_numpy(np.ascontiguousarray(shape_hw))

    dt = boxes.dtype
    batch = boxes.shape[0]
    # shape_hw is contractually 2-D (batch, 3) -- index it directly instead of
    # reshape(batch, -1): torch cannot resolve the -1 when batch == 0, and batch == 0
    # is a shape the host tiling explicitly accepts (empty-tensor fast path).
    hw = shape_hw.to(torch.int32).to(torch.float32)
    h = hw[:, 0]
    w = hw[:, 1]
    div4 = torch.stack([h, w, h, w], dim=1).to(dt)
    bx = boxes
    rank = bx.ndim
    if not reversed_box:
        div_shape = (batch,) + (1,) * (rank - 2) + (4,)
    else:
        div_shape = (batch, 4) + (1,) * (rank - 2)
    divisor = div4.reshape(div_shape)
    y = bx / divisor
    return y.to(dt)


def normalize_bbox_golden(boxes, shape_hw, reversed_box=False, **kwargs):
    """
    Golden function for normalize_bbox.
    All the parameters (names and order) follow @normalize_bbox_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    rb = bool(reversed_box)
    for k in ("reversedBox", "reversed"):
        if k in kwargs and kwargs[k] is not None:
            rb = bool(kwargs[k])
    result = _compute(boxes, shape_hw, rb)
    if not hasattr(boxes, "detach"):
        return result.numpy()
    return result
