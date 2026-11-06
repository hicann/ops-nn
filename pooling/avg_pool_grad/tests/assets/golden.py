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


__golden__ = {"kernel": {"avg_pool_grad": "avg_pool_grad_golden"}}


def due_fp16_overflow(data):
    """Overflow interception"""
    data = np.maximum(data, -65504)
    data = np.minimum(data, 65504)
    return data


def avg_pool_grad_golden(
    orig_input_shape, input_grad, ksize, strides, padding, data_format="NHWC", **kwargs
):
    """
    Kernel golden for avg_pool_grad.
    All the parameters follow @avg_pool_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.
    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
        input_formats, output_formats, input_ori_formats, output_ori_formats,
        input_dtypes, output_dtypes.
    """
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    output_dtype = kwargs.get("output_dtypes")

    # Collect shape info
    print("[WARNING] AvgpoolGrad golden func")
    h_index = data_format.index("H")
    w_index = data_format.index("W")

    def _spatial_pair(param):
        if isinstance(param, int):
            return param, param
        if len(param) == 2:
            return param[0], param[1]
        return param[h_index], param[w_index]

    strideh, stridew = _spatial_pair(strides)
    ksize_h, ksize_w = _spatial_pair(ksize)
    orig_input_shape = [int(v) for v in orig_input_shape]
    if data_format == "NCHW":
        N, C, H, W = orig_input_shape
        grad_nhwc = input_grad.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        N, H, W, C = orig_input_shape
        grad_nhwc = input_grad.astype(np.float32)
    orig_input_shape_nhwc = (N, H, W, C)
    tensor_dy = tf.compat.v1.placeholder(shape=grad_nhwc.shape, dtype=np.float32)
    avgpoolgrad_result = tf.compat.v1.raw_ops.AvgPoolGrad(
        orig_input_shape=orig_input_shape_nhwc,
        grad=tensor_dy,
        ksize=[1, ksize_h, ksize_w, 1],
        strides=[1, strideh, stridew, 1],
        padding=padding,
        data_format="NHWC",
        name="avg_pool_grad",
    )
    feed_dict = {tensor_dy: grad_nhwc}
    init_op = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        # Generate output tf data
        out = sess.run(avgpoolgrad_result, feed_dict=feed_dict)
    if data_format == "NCHW":
        output = out.transpose(0, 3, 1, 2)
    else:
        output = out
    if output_dtype[0] == "float16":
        output = due_fp16_overflow(output.astype(np.float16, copy=False))
    return output
