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

__golden__ = {"kernel": {"gather_nd": "gather_nd_golden"}}


def gather_nd_golden(params, indices, **kwargs):
    '''
    Golden function for gather_nd.
    All the parameters (names and order) follow @gather_nd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    params_shape = params.shape
    indices_shape = indices.shape

    data_dtype = params.dtype
    if data_dtype.name == 'bfloat16':
        params = params.view('int16')

    params_ph = tf.compat.v1.placeholder(dtype=params.dtype.name, shape=params_shape)
    indices_ph = tf.compat.v1.placeholder(dtype=indices.dtype.name, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather_nd(params_ph, indices_ph, name=None, batch_dims=0)
        res = sess.run(gather_res, feed_dict={params_ph: params, indices_ph: indices})

    if data_dtype.name == 'bfloat16':
        res = res.view(data_dtype)

    return res
