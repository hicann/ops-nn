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

__golden__ = {"kernel": {"max_pool_v2": "max_pool_v2_golden"}}


def max_pool_v2_golden(x, ksize, strides, *, padding, data_format="NHWC", **kwargs):
    """
    Golden function for max_pool_v2.
    All the parameters (names and order) follow @max_pool_v2_def.cpp without outputs.
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

    ksize = np.array(ksize).tolist()
    strides = np.array(strides).tolist()

    if data_format.lower() not in ["nchw", "nhwc"]:
        raise Exception("MaxPoolV2 only support NHWC and NCHW")
    if padding not in ["SAME", "VALID"]:
        raise Exception("MaxPoolV2 padding only support SAME and VALID")

    if data_format == "NCHW":
        inputx = np.transpose(inputx, (0, 2, 3, 1))
        ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
        strides = [strides[0], strides[2], strides[3], strides[1]]

    inputTensor = tf.compat.v1.placeholder(shape=inputx.shape, dtype=inputx.dtype)
    max_pool = tf.nn.max_pool2d(
        inputTensor,
        ksize=ksize,
        strides=strides,
        padding=padding,
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
