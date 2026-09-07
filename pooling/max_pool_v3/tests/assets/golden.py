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

__golden__ = {
    "aclnn": {
        "aclnnMaxPool": "aclnn_max_pool_golden",
    },
    "kernel": {"max_pool_v3": "max_pool_v3_golden"},
}


def max_pool_v3_golden(
    x,
    *,
    ksize,
    strides,
    padding_mode="CALCULATED",
    pads=[0, 0, 0, 0],
    data_format="NCHW",
    global_pooling=False,
    ceil_mode=False,
    **kwargs,
):
    """
    Golden function for max_pool_v3.
    All the parameters (names and order) follow @max_pool_v3_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    inputx = x
    input_dtype = inputx.dtype
    if "float16" in str(input_dtype):
        inputx = inputx.astype(np.float32)

    if data_format.lower() not in ["nchw", "nhwc"]:
        raise Exception("MaxPoolV3 only support NHWC and NCHW")
    if data_format == "NCHW":
        inputx = np.transpose(inputx, (0, 2, 3, 1))
        ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
        strides = [strides[0], strides[2], strides[3], strides[1]]

    if global_pooling:
        ksize = list(inputx.shape)
        ksize[0] = 1
        ksize[-1] = 1
        strides = [1, 1, 1, 1]
        padding_mode = "VALID"
    elif padding_mode == "CALCULATED":
        input_b_pad = pads[1]
        input_r_pad = pads[3]
        if ceil_mode:
            h_pad = 0
            input_b_pad = pads[1]
            out_h_floor = (inputx.shape[1] + pads[0] + pads[1] - ksize[1]) // strides[
                1
            ] + 1
            out_h_ceil = (
                inputx.shape[1] + pads[0] + pads[1] - ksize[1] + strides[1] - 1
            ) // strides[1] + 1
            if (out_h_ceil - 1) * strides[1] >= inputx.shape[1] + pads[0]:
                out_h_ceil -= 1
            if out_h_ceil > out_h_floor:
                h_pad = (
                    ((out_h_ceil - 1) * strides[1] + ksize[1])
                    - inputx.shape[1]
                    - pads[0]
                )
                input_b_pad = 0

            w_pad = 0
            out_w_floor = (inputx.shape[2] + pads[2] + pads[3] - ksize[2]) // strides[
                2
            ] + 1
            out_w_ceil = (
                inputx.shape[2] + pads[2] + pads[3] - ksize[2] + strides[2] - 1
            ) // strides[2] + 1
            if (out_w_ceil - 1) * strides[2] >= inputx.shape[2] + pads[2]:
                out_w_ceil -= 1
            if out_w_ceil > out_w_floor:
                w_pad = (
                    ((out_w_ceil - 1) * strides[2] + ksize[2])
                    - inputx.shape[2]
                    - pads[2]
                )
                input_r_pad = 0

            inputx = np.pad(
                inputx,
                ((0, 0), (0, h_pad), (0, w_pad), (0, 0)),
                mode="constant",
                constant_values=np.min(inputx),
            )
        pads = [[0, 0], [pads[0], input_b_pad], [pads[2], input_r_pad], [0, 0]]

        padding_mode = pads

    inputTensor = tf.compat.v1.placeholder(shape=inputx.shape, dtype=inputx.dtype)
    max_pool = tf.nn.max_pool2d(
        inputTensor,
        ksize=ksize,
        strides=strides,
        padding=padding_mode,
        data_format="NHWC",
        name="max_pool",
    )
    feed_dict = {inputTensor: inputx}

    with tf.compat.v1.Session() as sess:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)
        result = sess.run(max_pool, feed_dict=feed_dict)
    if data_format == "NCHW":
        result = np.transpose(result, (0, 3, 1, 2))
    return result.astype(input_dtype, copy=False)


def aclnn_max_pool_golden(
    self,
    kernelShape=0,
    strides=0,
    autoPad=0,
    pads=0,
    dilations=0,
    ceilMode=0,
    out=None,
    **kwargs,
):
    """
    Aclnn golden for aclnnMaxPool.
    Parameters follow @aclnnMaxPoolGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    x = self
    attrs = kwargs.get("attributes", {})
    kernel_size = attrs.get("kernelShape", kernelShape) if attrs else kernelShape
    stride = attrs.get("strides", strides) if attrs else strides
    padding = attrs.get("pads", pads) if attrs else pads
    dilation = attrs.get("dilations", dilations) if attrs else dilations
    ceil_mode = attrs.get("ceilMode", ceilMode) if attrs else ceilMode
    tensor_format = kwargs.get("tensor_formats", ["NCHW"])[0]

    is_nhwc = False
    if tensor_format.lower() == "nhwc":
        x = torch.permute(x, dims=[0, 2, 3, 1])
        is_nhwc = True

    input_dtype = x.dtype
    if "float16" in str(input_dtype):
        x = x.to(torch.float32)

    res = torch.nn.functional.max_pool2d(
        x,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=bool(ceil_mode) if ceil_mode else False,
        return_indices=False,
    )

    res = res.to(input_dtype)
    if is_nhwc:
        res = torch.permute(res, dims=[0, 3, 1, 2])
    return res
