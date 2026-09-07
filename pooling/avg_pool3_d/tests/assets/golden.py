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
import torch

__golden__ = {
    "aclnn": {
        "aclnnAvgPool3d": "aclnn_avg_pool3d_golden",
    },
    "kernel": {"avg_pool3_d": "avg_pool3_d_golden"},
}


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


def avg_pool3_d_golden(
    x,
    *,
    ksize,
    strides,
    pads,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=0,
    data_format="NDHWC",
    **kwargs,
):
    """
    Golden function for avg_pool3_d.
    All the parameters (names and order) follow @avg_pool3_d_def.cpp without outputs.
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

    if divisor_override == 0:
        divisor_override = None

    if len(strides) == 1:
        strides = [strides[0], strides[0], strides[0]]

    if len(ksize) == 1:
        ksize = [ksize[0], ksize[0], ksize[0]]

    if (
        len(pads) == 6
        and pads[0] == pads[1]
        and pads[2] == pads[3]
        and pads[4] == pads[5]
    ):
        pads = [pads[0], pads[2], pads[4]]
    if data_format.lower() not in ["ncdhw", "ndhwc"]:
        raise Exception("MaxPoolV3 only support NDHWC and NCDHW")

    if data_format == "NDHWC":
        inputx = np.transpose(inputx, (0, 4, 1, 2, 3))
        if len(ksize) == 5:
            ksize = [ksize[1], ksize[2], ksize[3]]
        if len(strides) == 5:
            strides = [strides[1], strides[2], strides[3]]
    else:
        if len(ksize) == 5:
            ksize = [ksize[2], ksize[3], ksize[4]]
        if len(strides) == 5:
            strides = [strides[2], strides[3], strides[4]]
    din, hin, win = inputx.shape[2:]
    out = None
    if (
        len(pads) == 1
        and (
            pads[0] < ksize[0] / 2 and pads[0] < ksize[1] / 2 and pads[0] < ksize[2] / 2
        )
    ) or (
        len(pads) == 3
        and (
            pads[0] < ksize[0] / 2 and pads[1] < ksize[1] / 2 and pads[2] < ksize[2] / 2
        )
    ):
        x_tensor = torch.tensor(inputx)
        out = torch.nn.functional.avg_pool3d(
            x_tensor,
            ksize,
            stride=strides,
            padding=pads,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
    else:
        if len(pads) == 1:
            pads = [pads[0], pads[0], pads[0], pads[0], pads[0], pads[0]]
        elif len(pads) == 3:
            pads = [pads[0], pads[0], pads[1], pads[1], pads[2], pads[2]]
        out_d, out_h, out_w = 0, 0, 0

        out_d = get_out_shape(
            inputx.shape[2], pads[0], pads[1], ksize[0], 1, strides[0], ceil_mode
        )
        out_h = get_out_shape(
            inputx.shape[3], pads[2], pads[3], ksize[1], 1, strides[1], ceil_mode
        )
        out_w = get_out_shape(
            inputx.shape[4], pads[4], pads[5], ksize[2], 1, strides[2], ceil_mode
        )
        org_pads = pads[:]

        in_d = (out_d - 1) * strides[0] + (1 * (ksize[0] - 1) + 1)
        in_h = (out_h - 1) * strides[1] + (1 * (ksize[1] - 1) + 1)
        in_w = (out_w - 1) * strides[2] + (1 * (ksize[2] - 1) + 1)
        pads[1] = max(in_d - inputx.shape[2] - pads[0], 0)
        pads[3] = max(in_h - inputx.shape[3] - pads[2], 0)
        pads[5] = max(in_w - inputx.shape[4] - pads[4], 0)

        inputx = np.pad(
            inputx,
            (
                (0, 0),
                (0, 0),
                (pads[0], pads[1]),
                (pads[2], pads[3]),
                (pads[4], pads[5]),
            ),
            mode="constant",
            constant_values=0,
        )
        x_tensor = torch.tensor(inputx)
        out = torch.nn.functional.avg_pool3d(
            x_tensor,
            ksize,
            stride=strides,
            padding=0,
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=1,
        )
        if (divisor_override is not None) and divisor_override != 0:
            out = out / torch.tensor(divisor_override, dtype=torch.float32)
        else:
            idx = torch.arange(out_d * out_h * out_w)
            d_idx = idx // (out_h * out_w)
            h_idx = idx % (out_h * out_w) // out_w
            w_idx = idx % out_w
            d_start = d_idx * strides[0] - org_pads[0]
            h_start = h_idx * strides[1] - org_pads[2]
            w_start = w_idx * strides[2] - org_pads[4]
            d_end = d_start + ksize[0]
            h_end = h_start + ksize[1]
            w_end = w_start + ksize[2]
            if count_include_pad:
                d_end.clamp_(max=din + org_pads[1])
                h_end.clamp_(max=hin + org_pads[3])
                w_end.clamp_(max=win + org_pads[5])
            else:
                d_start.clamp_(min=0)
                h_start.clamp_(min=0)
                w_start.clamp_(min=0)
                d_end.clamp_(max=din)
                h_end.clamp_(max=hin)
                w_end.clamp_(max=win)
            window_d = d_end - d_start
            window_h = h_end - h_start
            window_w = w_end - w_start
            pool_size = window_d * window_h * window_w
            pool_size = pool_size.to(torch.float).reshape(1, 1, out_d, out_h, out_w)
            out = out / pool_size
    res = out.numpy()
    if data_format == "NDHWC":
        res = np.transpose(res, (0, 2, 3, 4, 1))
    res = res.astype(input_dtype, copy=False)
    return res


def aclnn_avg_pool3d_golden(
    self,
    kernelSize=0,
    stride=0,
    padding=0,
    ceilMode=0,
    countIncludePad=0,
    divisorOverride=0,
    out=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnAvgPool3d.
    Parameters follow @aclnnAvgPool3dGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    inputx = self
    input_dtype = inputx.dtype

    if "float16" in str(input_dtype):
        inputx = inputx.to(torch.float32)
    attrs = kwargs.get("attributes", {})
    ksize = attrs.get("kernelSize", kernelSize)
    strides = attrs.get("stride", stride)
    pads = attrs.get("padding", padding if padding else [0, 0, 0, 0, 0, 0])
    ceil_mode = attrs.get("ceilMode", ceilMode)
    countIncludePad = attrs.get("countIncludePad", countIncludePad)
    divisor_override = attrs.get("divisorOverride", divisorOverride)
    data_format = kwargs.get("tensor_formats", ["NCDHW"])[0]
    if divisor_override == 0:
        divisor_override = None

    if len(strides) == 1:
        strides = [strides[0], strides[0], strides[0]]

    if len(ksize) == 1:
        ksize = [ksize[0], ksize[0], ksize[0]]

    if (
        len(pads) == 6
        and pads[0] == pads[1]
        and pads[2] == pads[3]
        and pads[4] == pads[5]
    ):
        pads = [pads[0], pads[2], pads[4]]

    if data_format.lower() not in ["ncdhw", "nd"]:
        raise Exception("AvgPool3d only support ND and NCDHW")

    if data_format == "ND":
        inputx = inputx.reshape(
            1, inputx.shape[0], inputx.shape[1], inputx.shape[2], inputx.shape[3]
        )
    if len(ksize) == 5:
        ksize = [ksize[2], ksize[3], ksize[4]]
    if len(strides) == 5:
        strides = [strides[2], strides[3], strides[4]]
    din, hin, win = inputx.shape[2:]
    out = None
    if (
        len(pads) == 1
        and (
            pads[0] < ksize[0] / 2 and pads[0] < ksize[1] / 2 and pads[0] < ksize[2] / 2
        )
    ) or (
        len(pads) == 3
        and (
            pads[0] < ksize[0] / 2 and pads[1] < ksize[1] / 2 and pads[2] < ksize[2] / 2
        )
    ):
        out = torch.nn.functional.avg_pool3d(
            inputx,
            ksize,
            stride=strides,
            padding=pads,
            ceil_mode=ceil_mode,
            count_include_pad=countIncludePad,
            divisor_override=divisor_override,
        )
    else:
        if len(pads) == 1:
            pads = [pads[0], pads[0], pads[0], pads[0], pads[0], pads[0]]
        elif len(pads) == 3:
            pads = [pads[0], pads[0], pads[1], pads[1], pads[2], pads[2]]
        out_d, out_h, out_w = 0, 0, 0

        out_d = get_out_shape(
            inputx.shape[2], pads[0], pads[1], ksize[0], 1, strides[0], ceil_mode
        )
        out_h = get_out_shape(
            inputx.shape[3], pads[2], pads[3], ksize[1], 1, strides[1], ceil_mode
        )
        out_w = get_out_shape(
            inputx.shape[4], pads[4], pads[5], ksize[2], 1, strides[2], ceil_mode
        )
        org_pads = pads[:]

        in_d = (out_d - 1) * strides[0] + (1 * (ksize[0] - 1) + 1)
        in_h = (out_h - 1) * strides[1] + (1 * (ksize[1] - 1) + 1)
        in_w = (out_w - 1) * strides[2] + (1 * (ksize[2] - 1) + 1)
        pads[1] = max(in_d - inputx.shape[2] - pads[0], 0)
        pads[3] = max(in_h - inputx.shape[3] - pads[2], 0)
        pads[5] = max(in_w - inputx.shape[4] - pads[4], 0)

        inputx = np.pad(
            inputx,
            (
                (0, 0),
                (0, 0),
                (pads[0], pads[1]),
                (pads[2], pads[3]),
                (pads[4], pads[5]),
            ),
            mode="constant",
            constant_values=0,
        )
        x_tensor = torch.tensor(inputx)
        out = torch.nn.functional.avg_pool3d(
            x_tensor,
            ksize,
            stride=strides,
            padding=0,
            ceil_mode=False,
            count_include_pad=False,
            divisor_override=1,
        )
        if (divisor_override is not None) and divisor_override != 0:
            out = out / torch.tensor(divisor_override, dtype=torch.float32)
        else:
            idx = torch.arange(out_d * out_h * out_w)
            d_idx = idx // (out_h * out_w)
            h_idx = idx % (out_h * out_w) // out_w
            w_idx = idx % out_w
            d_start = d_idx * strides[0] - org_pads[0]
            h_start = h_idx * strides[1] - org_pads[2]
            w_start = w_idx * strides[2] - org_pads[4]
            d_end = d_start + ksize[0]
            h_end = h_start + ksize[1]
            w_end = w_start + ksize[2]
            if countIncludePad:
                d_end.clamp_(max=din + org_pads[1])
                h_end.clamp_(max=hin + org_pads[3])
                w_end.clamp_(max=win + org_pads[5])
            else:
                d_start.clamp_(min=0)
                h_start.clamp_(min=0)
                w_start.clamp_(min=0)
                d_end.clamp_(max=din)
                h_end.clamp_(max=hin)
                w_end.clamp_(max=win)
            window_d = d_end - d_start
            window_h = h_end - h_start
            window_w = w_end - w_start
            pool_size = window_d * window_h * window_w
            pool_size = pool_size.to(torch.float).reshape(1, 1, out_d, out_h, out_w)
            out = out / pool_size
    if data_format == "ND":
        out = out.reshape(out.shape[1], out.shape[2], out.shape[3], out.shape[4])
    res = out.to(input_dtype)
    return res
