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


__golden__ = {
    "aclnn": {
        "aclnnGatherV2": "aclnn_gather_v2_golden",
    },
    "kernel": {"gather_v2": "gather_v2_golden"},
}


def gather_v2_golden(
    x, indices, axis, *, batch_dims=0, negative_index_support=False, **kwargs
):
    """
    Golden function for gather_v2.
    All the parameters (names and order) follow @gather_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    params_data = x
    params_shape_len = len(params_data.shape)

    indices_data = indices
    indices_shape_len = len(indices_data.shape)

    batch_dims = batch_dims if batch_dims >= 0 else batch_dims + indices_shape_len

    axis = int(axis)
    axis = axis if axis >= 0 else axis + params_shape_len

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    params_shape = params_data.shape
    indices_shape = indices_data.shape

    data_dtype = params_data.dtype
    if data_dtype.name == "bfloat16":
        params_data = params_data.view("int16")

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(
            params, indices, axis=axis, batch_dims=batch_dims, name=None
        )
        res = sess.run(
            gather_res, feed_dict={params: params_data, indices: indices_data}
        )

    if data_dtype.name == "bfloat16":
        res = res.view(data_dtype)

    return res


def aclnn_gather_v2_golden(self, dim, index, out=None, **kwargs):
    """
    Aclnn golden for aclnnGatherV2.
    """
    import tensorflow as tf
    import torch

    tensor_x = self
    x_dtype = self.dtype
    if "bfloat16" in str(x_dtype):
        tensor_x = self.to(torch.float32)

    if hasattr(dim, "item"):
        dim = dim.item()

    tf_out = tf.gather(tensor_x, index, axis=dim)
    np_out = tf_out.numpy()
    pt_out = torch.from_numpy(np_out)
    return pt_out
