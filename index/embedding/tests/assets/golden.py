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
        "aclnnEmbedding": "aclnn_embedding_golden",
    },
    "kernel": {"embedding": "embedding_golden"},
}


def embedding_golden(x, indices, **kwargs):
    """
    Golden function for embedding.
    All the parameters (names and order) follow @embedding_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    params_data = x

    indices_data = indices

    batch_dims = 0
    axis = 0

    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()

    params_shape = params_data.shape
    indices_shape = indices_data.shape

    params = tf.compat.v1.placeholder(dtype=params_data.dtype, shape=params_shape)
    indices = tf.compat.v1.placeholder(dtype=indices_data.dtype, shape=indices_shape)

    with tf.compat.v1.Session() as sess:
        gather_res = tf.compat.v1.gather(
            params, indices, axis=axis, batch_dims=batch_dims, name=None
        )
        res = sess.run(
            gather_res, feed_dict={params: params_data, indices: indices_data}
        )

    return res


def aclnn_embedding_golden(weight, indices, out=None, **kwargs):
    """
    Aclnn golden for aclnnEmbedding.
    Parameters follow @aclnnEmbeddingGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    return torch.nn.functional.embedding(indices, weight)
