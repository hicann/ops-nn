#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np

__golden__ = {"kernel": {"max_pool3d_grad": "max_pool3d_grad_golden"}}


# __input__ = {"kernel": {"max_pool3d_grad": "customize_inputs"}}
def _forward_max_pool3d(orig_x, ksize, strides, padding, pads, data_format):
    x = orig_x.astype(np.float32)
    x_shape = x.shape
    if data_format == "NDHWC":
        n, d, h, w, c = x_shape
        kd, kh, kw = ksize[1], ksize[2], ksize[3]
        sd, sh, sw = strides[1], strides[2], strides[3]
        x = np.transpose(x, (0, 4, 1, 2, 3))
    elif data_format == "NCDHW":
        n, c, d, h, w = x_shape
        kd, kh, kw = ksize[2], ksize[3], ksize[4]
        sd, sh, sw = strides[2], strides[3], strides[4]
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
    if padding == "SAME":
        yd = (d + sd - 1) // sd
        yh = (h + sh - 1) // sh
        yw = (w + sw - 1) // sw
        pad_d = max((yd - 1) * sd + kd - d, 0)
        pad_h = max((yh - 1) * sh + kh - h, 0)
        pad_w = max((yw - 1) * sw + kw - w, 0)
        pf, pb = pad_d // 2, pad_d - pad_d // 2
        pt, pbo = pad_h // 2, pad_h - pad_h // 2
        pl, pr = pad_w // 2, pad_w - pad_w // 2
    elif padding == "VALID":
        yd = (d - kd) // sd + 1
        yh = (h - kh) // sh + 1
        yw = (w - kw) // sw + 1
        pf = pb = pt = pbo = pl = pr = 0
    else:
        if pads is not None and len(pads) == 6:
            pf, pb, pt, pbo, pl, pr = pads
        elif pads is not None and len(pads) == 3:
            pf, pt, pl = pads
            pb, pbo, pr = pads
        else:
            pf = pb = pt = pbo = pl = pr = 0
        yd = (d + pf + pb - kd) // sd + 1
        yh = (h + pt + pbo - kh) // sh + 1
        yw = (w + pl + pr - kw) // sw + 1
    if pf or pb or pt or pbo or pl or pr:
        x = np.pad(
            x, ((0, 0), (0, 0), (pf, pb), (pt, pbo), (pl, pr)), constant_values=-np.inf
        )
    y = np.zeros((n, c, yd, yh, yw), dtype=np.float32)
    for nn in range(n):
        for cc in range(c):
            for od in range(yd):
                for oh in range(yh):
                    for ow in range(yw):
                        ds, hs, ws = od * sd, oh * sh, ow * sw
                        window = x[nn, cc, ds : ds + kd, hs : hs + kh, ws : ws + kw]
                        y[nn, cc, od, oh, ow] = window.max()
    if data_format == "NDHWC":
        y = np.transpose(y, (0, 2, 3, 4, 1))
    return y.astype(orig_x.dtype)


def customize_inputs(
    orig_x,
    orig_y,
    grads,
    *,
    ksize,
    strides,
    padding="SAME",
    pads=None,
    data_format="NDHWC",
    **kwargs,
):
    orig_x = (
        np.arange(orig_x.size, dtype=np.float32)
        .reshape(orig_x.shape)
        .astype(orig_x.dtype)
    )
    grads = (
        np.arange(grads.size, dtype=np.float32).reshape(grads.shape).astype(grads.dtype)
    )
    orig_y = _forward_max_pool3d(orig_x, ksize, strides, padding, pads, data_format)
    return (orig_x, orig_y, grads)


def max_pool3d_grad_golden(
    orig_x,
    orig_y,
    grads,
    *,
    ksize,
    strides,
    padding="SAME",
    pads=None,
    data_format="NDHWC",
    **kwargs,
):
    """
    Golden function for max_pool3d_grad.
    All the parameters (names and order) follow @max_pool3d_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name
    Returns:
        Output tensor
    """
    input_dtype = orig_x.dtype
    # ml_dtypes.bfloat16 直接 astype 即完成数值转换; 禁用 view(int16) 位模式口径
    x = orig_x.astype(np.float32)
    y = orig_y.astype(np.float32)
    g = grads.astype(np.float32)
    x_shape = x.shape
    y_shape = y.shape
    if data_format == "NDHWC":
        n, d, h, w, c = x_shape
        yn, yd, yh, yw, yc = y_shape
        x = np.transpose(x, (0, 4, 1, 2, 3))
        y = np.transpose(y, (0, 4, 1, 2, 3))
        g = np.transpose(g, (0, 4, 1, 2, 3))
    elif data_format == "NCDHW":
        n, c, d, h, w = x_shape
        yn, yc, yd, yh, yw = y_shape
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")
    if len(ksize) == 5:
        if data_format == "NDHWC":
            kd, kh, kw = ksize[1], ksize[2], ksize[3]
            sd, sh, sw = strides[1], strides[2], strides[3]
        else:
            kd, kh, kw = ksize[2], ksize[3], ksize[4]
            sd, sh, sw = strides[2], strides[3], strides[4]
    else:
        kd, kh, kw = ksize[0], ksize[1], ksize[2]
        sd, sh, sw = strides[0], strides[1], strides[2]
    if padding == "SAME":
        pad_d = max((yd - 1) * sd + kd - d, 0)
        pad_h = max((yh - 1) * sh + kh - h, 0)
        pad_w = max((yw - 1) * sw + kw - w, 0)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padding_tuple = (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right)
    elif padding == "VALID":
        padding_tuple = (0, 0, 0, 0, 0, 0)
    else:
        if pads is not None and len(pads) == 6:
            padding_tuple = (pads[0], pads[1], pads[2], pads[3], pads[4], pads[5])
        elif pads is not None and len(pads) == 3:
            padding_tuple = (pads[0], pads[0], pads[1], pads[1], pads[2], pads[2])
        else:
            padding_tuple = (0, 0, 0, 0, 0, 0)
    pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right = padding_tuple
    grad_output = np.zeros((n, c, d, h, w), dtype=np.float32)
    flat_grads = g.reshape(n, c, -1)
    for nn in range(n):
        for cc in range(c):
            for pd in range(yd):
                for ph in range(yh):
                    for pw in range(yw):
                        d_start = pd * sd - pad_front
                        h_start = ph * sh - pad_top
                        w_start = pw * sw - pad_left
                        max_val = -np.inf
                        max_d_in, max_h_in, max_w_in = -1, -1, -1
                        for dd in range(kd):
                            for dh in range(kh):
                                for dw in range(kw):
                                    d_in = d_start + dd
                                    h_in = h_start + dh
                                    w_in = w_start + dw
                                    if (
                                        0 <= d_in < d
                                        and 0 <= h_in < h
                                        and 0 <= w_in < w
                                    ):
                                        val = x[nn, cc, d_in, h_in, w_in]
                                        # 首个界内元素无条件成为初值 argmax（对齐 kernel 的 origin 初值:
                                        # 全 -inf 窗口 argmax=首元素, grad 照常散射）; NaN 与 kernel 的
                                        # NE 接替语义一致（NaN 赢得 max）
                                        if (
                                            max_d_in < 0
                                            or val > max_val
                                            or np.isnan(val)
                                        ):
                                            max_val = val
                                            max_d_in, max_h_in, max_w_in = (
                                                d_in,
                                                h_in,
                                                w_in,
                                            )
                        if max_d_in >= 0:
                            grad_val = flat_grads[nn, cc, pd * yh * yw + ph * yw + pw]
                            # 重叠窗口可散射到同一 argmax 位置, 必须累加（对齐 torch aten 语义）
                            grad_output[nn, cc, max_d_in, max_h_in, max_w_in] += (
                                grad_val
                            )
    if data_format == "NDHWC":
        grad_output = np.transpose(grad_output, (0, 2, 3, 4, 1))
    result = grad_output
    if input_dtype.name == "bfloat16":
        # ml_dtypes.bfloat16 直接转换; transpose 后的非连续数组禁用 view() 位操作
        return result.astype(input_dtype)
    elif input_dtype == np.float16:
        return result.astype(np.float16, copy=False)
    else:
        return result.astype(input_dtype, copy=False)
