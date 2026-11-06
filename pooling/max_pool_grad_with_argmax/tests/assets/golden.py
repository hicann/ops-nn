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
    "kernel": {"max_pool_grad_with_argmax": "max_pool_grad_with_argmax_golden"}
}


def max_pool_grad_with_argmax_golden(
    x,
    grad,
    argmax,
    *,
    ksize,
    strides,
    padding,
    include_batch_in_index=False,
    data_format="NHWC",
    **kwargs,
):
    """
    Golden function for max_pool_grad_with_argmax.
    All the parameters (names and order) follow @max_pool_grad_with_argmax_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    ksizes = ksize
    strides = strides
    padding = padding
    attr_op_format = data_format

    def _parse_spatial_param(param, data_format):
        if isinstance(param, int):
            return param, param
        elif len(param) == 2:
            return param[0], param[1]
        elif len(param) == 4:
            if data_format == "NCHW":
                return param[2], param[3]
            else:
                return param[1], param[2]
        else:
            raise ValueError(f"Unsupported param format: {param}")

    kH, kW = _parse_spatial_param(ksizes, attr_op_format)
    sH, sW = _parse_spatial_param(strides, attr_op_format)

    tf_ksize = [1, kH, kW, 1]
    tf_strides = [1, sH, sW, 1]

    input_data = x
    grad_data = grad
    argmax_data = argmax

    if attr_op_format == "NCHW":
        N, C, H, W = input_data.shape
        N_g, C_g, H_out, W_out = grad_data.shape
        assert N_g == N and C_g == C, "grad shape mismatch"

        argmax_nchw = argmax_data
        argmax_nhwc = np.zeros((N, H_out, W_out, C), dtype=np.int64)

        for n in range(N):
            for c in range(C):
                for ph in range(H_out):
                    for pw in range(W_out):
                        nchw_index = argmax_nchw[n, c, ph, pw]
                        c_in = nchw_index // (H * W)
                        remainder = nchw_index % (H * W)
                        h_in = remainder // W
                        w_in = remainder % W

                        assert c_in == c, f"Channel mismatch in argmax: {c_in} vs {c}"

                        nhwc_index = h_in * (W * C) + w_in * C + c_in
                        argmax_nhwc[n, ph, pw, c] = nhwc_index

        input_nhwc = input_data.transpose(0, 2, 3, 1)
        grad_nhwc = grad_data.transpose(0, 2, 3, 1)

        maxpoolgrad_res = tf.raw_ops.MaxPoolGradWithArgmax(
            input=input_nhwc,
            grad=grad_nhwc,
            argmax=argmax_nhwc,
            ksize=tf_ksize,
            strides=tf_strides,
            padding=padding,
            include_batch_in_index=include_batch_in_index,
            name="maxpoolgradwithargmax",
        )

        with tf.compat.v1.Session() as sess:
            res_nhwc = sess.run(maxpoolgrad_res)

        res = res_nhwc.transpose(0, 3, 1, 2)

    else:
        argmax_data = argmax_data.astype(np.int64)
        maxpoolgrad_res = tf.raw_ops.MaxPoolGradWithArgmax(
            input=input_data,
            grad=grad_data,
            argmax=argmax_data,
            ksize=tf_ksize,
            strides=tf_strides,
            padding=padding,
            include_batch_in_index=include_batch_in_index,
        )
        with tf.compat.v1.Session() as sess:
            res = sess.run(maxpoolgrad_res)

    return res.astype(kwargs["output_dtypes"][0], copy=False)
