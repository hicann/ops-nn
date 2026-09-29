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
        "aclnnIndexSelect": "aclnn_gather_v2_golden",
    },
    "kernel": {"gather_v2": "gather_v2_golden"},
    "e2e": {"torch.Tensor.index_select": "aclnn_gather_v2_golden"},
}


def empty_tensor_reshape(param):
    import torch

    if isinstance(param, torch.Tensor):
        if param.dim() == 0:
            param = param.reshape(1)
    else:
        if param.ndim == 0:
            param = param.reshape(1)
    return param


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
    x_dtype_str = kwargs.get("input_dtypes", ["float16"])[0]
    is_complex32 = "complex32" in str(x_dtype_str)
    params_shape_len = (
        (len(params_data.shape) - 1) if is_complex32 else len(params_data.shape)
    )

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
    dtype_size = params_data.itemsize
    if data_dtype.name == "bfloat16":
        params_data = params_data.view("int16")
    elif dtype_size == 1:
        params_data = params_data.view("int8")

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(
            params, indices, axis=axis, batch_dims=batch_dims, name=None
        )
        res = sess.run(
            gather_res, feed_dict={params: params_data, indices: indices_data}
        )

    if data_dtype.name == "bfloat16" or dtype_size == 1:
        res = res.view(data_dtype)

    return res


def aclnn_gather_v2_golden(self, dim, index, out=None, **kwargs):
    """
    Aclnn golden for aclnnGatherV2 / aclnnIndexSelect.
    Also serves as the e2e golden for torch.Tensor.index_select:
    the e2e plugin calls it with the same (self, dim, index) plan,
    and torch.Tensor.index_select shares aclnnIndexSelect semantics
    (index is 1-D, output = tf.gather(self, index, axis=dim)).
    """
    import numpy as np
    import tensorflow as tf
    import torch

    x_dtype_str = str(self.dtype)
    is_hifloat8 = "hifloat8" in x_dtype_str

    # 空tensor处理
    self = empty_tensor_reshape(self)
    index = empty_tensor_reshape(index)

    tensor_x = self
    if is_hifloat8:
        tensor_x = self.view(np.uint8)
    elif "bfloat16" in x_dtype_str:
        tensor_x = self.view(torch.int16)

    if hasattr(dim, "item"):
        dim = dim.item()

    tf_out = tf.gather(tensor_x, index, axis=dim)
    np_out = tf_out.numpy()
    if (
        is_hifloat8 and "backend" not in kwargs
    ):  # aclnn场景需要把uint8转回hif8，e2e场景不需要
        # torch 无法表示 hifloat8：按位还原为输入的 hifloat8 dtype，以 numpy 数组返回
        return np_out.view(self.dtype)
    pt_out = torch.from_numpy(np_out)
    if "bfloat16" in x_dtype_str:
        pt_out = pt_out.view(torch.bfloat16)
    return pt_out
