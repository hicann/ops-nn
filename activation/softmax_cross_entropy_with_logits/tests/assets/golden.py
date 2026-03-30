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

__golden__ = {"kernel": {"softmax_cross_entropy_with_logits": "softmax_cross_entropy_with_logits_golden"}}


def softmax_cross_entropy_with_logits_golden(features, labels, **kwargs):
    '''
    Golden function for softmax_cross_entropy_with_logits.
    All the parameters (names and order) follow @softmax_cross_entropy_with_logits_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    from tensorflow.python.ops import gen_nn_ops
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    input_dtype = features.dtype
    data1 = features
    data2 = labels
    if input_dtype.name == "bfloat16" or input_dtype.name == "float16":
        data1 = data1.astype("float32")
        data2 = data2.astype("float32")

    input1 = tf.compat.v1.placeholder(shape=data1.shape, dtype=data1.dtype)
    input2 = tf.compat.v1.placeholder(shape=data2.shape, dtype=data2.dtype)
    output_data = gen_nn_ops.softmax_cross_entropy_with_logits(input1, input2)
    feed_dict = {input1: data1, input2: data2}

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        result1, result2 = sess.run(output_data, feed_dict=feed_dict)

    return [result1.astype(input_dtype, copy=False), result2.astype(input_dtype, copy=False)]
