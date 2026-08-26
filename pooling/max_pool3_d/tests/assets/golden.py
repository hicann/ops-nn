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

__golden__ = {"kernel": {"max_pool3_d": "max_pool3_d_golden"}}


def get_out_shape(in_width, pad_left, pad_right, kw, dilation, stride, ceil_mode=False):
    out_width = 0
    if ceil_mode:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1) + stride - 1
        ) // stride + 1
        if (out_width - 1) * stride >= in_width + pad_left:
            out_width = out_width - 1
    else:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1)
        ) // stride + 1
    return out_width


def get_pad_for_same(out_width, stride, kw, dilation, in_width):
    pad_need = max((out_width - 1) * stride + ((kw - 1) * dilation + 1) - in_width, 0)
    pad_left = pad_need // 2
    pad_right = pad_need - pad_left
    return pad_left, pad_right


def max_pool3_d_golden(
    x,
    *,
    ksize,
    strides,
    padding,
    pads=None,
    dilation=[1, 1, 1, 1, 1],
    ceil_mode=0,
    data_format="NDHWC",
    **kwargs,
):
    """
    Golden function for max_pool3_d.
    All the parameters (names and order) follow @max_pool3_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import torch

    inputx = x
    input_dtype = inputx.dtype

    if "float16" in str(input_dtype):
        inputx = inputx.astype(np.float32)
    padding_mode = padding
    if pads is None:
        pads = [0, 0, 0, 0, 0, 0]
    pads = list(pads)
    if len(pads) == 1:
        pads = [pads[0], pads[0], pads[0], pads[0], pads[0], pads[0]]
    elif len(pads) == 3:
        pads = [pads[0], pads[0], pads[1], pads[1], pads[2], pads[2]]

    if len(strides) == 1:
        strides = [strides[0], strides[0], strides[0]]

    if len(ksize) == 1:
        ksize = [ksize[0], ksize[0], ksize[0]]

    if len(dilation) == 1:
        dilation = [dilation[0], dilation[0], dilation[0]]

    if data_format.lower() not in ["ncdhw", "ndhwc"]:
        raise Exception("MaxPoolV3 only support NDHWC and NCDHW")
    if data_format == "NDHWC":
        inputx = np.transpose(inputx, (0, 4, 1, 2, 3))
        if len(ksize) == 5:
            ksize = [ksize[1], ksize[2], ksize[3]]
        if len(strides) == 5:
            strides = [strides[1], strides[2], strides[3]]
        if len(dilation) == 5:
            dilation = [dilation[1], dilation[2], dilation[3]]
    else:
        if len(ksize) == 5:
            ksize = [ksize[2], ksize[3], ksize[4]]
        if len(strides) == 5:
            strides = [strides[2], strides[3], strides[4]]
        if len(dilation) == 5:
            dilation = [dilation[2], dilation[3], dilation[4]]
    out_d, out_h, out_w = 0, 0, 0
    if padding_mode == "CALCULATED":
        out_d = get_out_shape(
            inputx.shape[2],
            pads[0],
            pads[1],
            ksize[0],
            dilation[0],
            strides[0],
            ceil_mode,
        )
        out_h = get_out_shape(
            inputx.shape[3],
            pads[2],
            pads[3],
            ksize[1],
            dilation[1],
            strides[1],
            ceil_mode,
        )
        out_w = get_out_shape(
            inputx.shape[4],
            pads[4],
            pads[5],
            ksize[2],
            dilation[2],
            strides[2],
            ceil_mode,
        )
    elif padding_mode == "VALID":
        out_d = get_out_shape(
            inputx.shape[2], 0, 0, ksize[0], dilation[0], strides[0], False
        )
        out_h = get_out_shape(
            inputx.shape[3], 0, 0, ksize[1], dilation[1], strides[1], False
        )
        out_w = get_out_shape(
            inputx.shape[4], 0, 0, ksize[2], dilation[2], strides[2], False
        )
        pads[0], pads[2], pads[4] = 0, 0, 0
    else:
        out_d = (inputx.shape[2] + strides[0] - 1) // strides[0]
        out_h = (inputx.shape[3] + strides[1] - 1) // strides[1]
        out_w = (inputx.shape[4] + strides[2] - 1) // strides[2]
        pads[0], _ = get_pad_for_same(
            out_d, strides[0], ksize[0], dilation[0], inputx.shape[2]
        )
        pads[2], _ = get_pad_for_same(
            out_h, strides[1], ksize[1], dilation[1], inputx.shape[3]
        )
        pads[4], _ = get_pad_for_same(
            out_w, strides[2], ksize[2], dilation[2], inputx.shape[4]
        )

    in_d = (out_d - 1) * strides[0] + (dilation[0] * (ksize[0] - 1) + 1)
    in_h = (out_h - 1) * strides[1] + (dilation[1] * (ksize[1] - 1) + 1)
    in_w = (out_w - 1) * strides[2] + (dilation[2] * (ksize[2] - 1) + 1)
    pads[1] = max(in_d - inputx.shape[2] - pads[0], 0)
    pads[3] = max(in_h - inputx.shape[3] - pads[2], 0)
    pads[5] = max(in_w - inputx.shape[4] - pads[4], 0)

    inputx = np.pad(
        inputx,
        ((0, 0), (0, 0), (pads[0], pads[1]), (pads[2], pads[3]), (pads[4], pads[5])),
        mode="constant",
        constant_values=(-np.inf),
    )
    x_tensor = torch.Tensor(inputx)

    out = torch.nn.functional.max_pool3d(
        x_tensor,
        ksize,
        stride=strides,
        padding=0,
        dilation=dilation,
        ceil_mode=False,
        return_indices=False,
    )
    res = out.numpy()
    if data_format == "NDHWC":
        res = np.transpose(res, (0, 2, 3, 4, 1))
    res = res.astype(input_dtype, copy=False)
    return res
