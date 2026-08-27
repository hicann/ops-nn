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
import zlib
import tensorflow as tf
from tensorflow.python.ops import array_ops

tf.compat.v1.disable_eager_execution()

__spec__ = {
    "broadcast_gradient_args": "BroadcastGradientArgsKernelSpec",
    "BroadcastGradientArgs": "BroadcastGradientArgsKernelSpec",
    "aclnnBroadcastGradientArgs": "BroadcastGradientArgsAclnnSpec",
}

__golden__ = {
    "kernel": {
        "broadcast_gradient_args": "broadcast_gradient_args_golden",
        "BroadcastGradientArgs": "broadcast_gradient_args_golden",
    },
    "aclnn": {
        "aclnnBroadcastGradientArgs": "aclnn_broadcast_gradient_args_golden",
    },
}

__input__ = {
    "kernel": {
        "broadcast_gradient_args": "broadcast_gradient_args_input",
        "BroadcastGradientArgs": "broadcast_gradient_args_input",
    }
}


def _to_numpy(t):
    if isinstance(t, np.ndarray):
        return t
    try:
        return t.numpy()
    except AttributeError:
        return np.asarray(t)


def _tf_broadcast_gradient_args(x1, x2):
    """
    调用 tensorflow array_ops.broadcast_gradient_args 计算 golden。
    x1/x2 为 numpy.ndarray，返回 (y1, y2) 两个 numpy.ndarray，
    dtype 与输入保持一致。
    """
    x1_tensor = tf.compat.v1.placeholder(shape=x1.shape, dtype=x1.dtype)
    x2_tensor = tf.compat.v1.placeholder(shape=x2.shape, dtype=x2.dtype)
    out = array_ops.broadcast_gradient_args(x1_tensor, x2_tensor)

    with tf.compat.v1.Session() as sess:
        res = sess.run(out, feed_dict={x1_tensor: x1, x2_tensor: x2})
    return res[0].astype(x1.dtype), res[1].astype(x2.dtype)


class BroadcastGradientArgsKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray"""

    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    def golden(x1, x2, **kwargs):
        return list(_tf_broadcast_gradient_args(x1, x2))


class BroadcastGradientArgsAclnnSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor（已在设备上）"""

    tolerance = {
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }

    def golden(x1, x2, y1, y2, **kwargs):
        return list(_tf_broadcast_gradient_args(_to_numpy(x1), _to_numpy(x2)))

    def gen_input(input_shapes, input_dtypes, input_ranges, **kwargs):
        """
        ACLNN 输入生成：保证 x1/x2 满足广播规则。
        input_shapes/dtypes/ranges 只包含纯输入张量（不含输出），
        顺序为 [x1, x2]。
        """
        x1_shape, x2_shape = input_shapes[0], input_shapes[1]
        x1_dtype, x2_dtype = input_dtypes[0], input_dtypes[1]

        testcase_name = kwargs.get("testcase_name", "")
        x1_tuple, x2_tuple = generate_broadcastable_shapes(
            x1_shape[0], x2_shape[0], testcase_name
        )
        x1_array = np.array(x1_tuple).astype(x1_dtype)
        x2_array = np.array(x2_tuple).astype(x2_dtype)
        return [x1_array, x2_array]


def broadcast_gradient_args_golden(x1, x2, **kwargs):
    """
    Kernel golden for broadcast_gradient_args.
    All the parameters (names and order) follow @broadcast_gradient_args_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    算子功能：在反向传播过程中，根据两个张量在正向传播时的原始形状，自动识别出它们因广播机制
    而扩展的维度，并输出需要在哪些维度上对梯度进行约简。

    Args:
        x1: np.ndarray, 原始张量a的shape，1维。
        x2: np.ndarray, 原始张量b的shape，1维。
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        tuple: (y1, y2)
            y1: np.ndarray, x1对应的张量shape中需要广播的索引。
            y2: np.ndarray, x2对应的张量shape中需要广播的索引。
    """
    return _tf_broadcast_gradient_args(x1, x2)


def aclnn_broadcast_gradient_args_golden(x1, x2, y1, y2, **kwargs):
    """
    Aclnn golden for aclnnBroadcastGradientArgs.
    All the parameters (name & order) follow
        function `aclnnBroadcastGradientArgsGetWorkspaceSize` in @aclnn_broadcast_gradient_args.h
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch,
        the Tensors in the parameters are all torch.Tensor.
    Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        x1: 输入张量，原始张量a的shape，1维。
        x2: 输入张量，原始张量b的shape，1维。
        y1: 输出张量（用于确定 dtype/shape），x1对应的广播索引。
        y2: 输出张量（用于确定 dtype/shape），x2对应的广播索引。
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        tuple: (y1_result, y2_result)
    """
    return _tf_broadcast_gradient_args(_to_numpy(x1), _to_numpy(x2))


def generate_broadcastable_shapes(ndim1: int, ndim2: int, testcase_name: str = ""):
    """
    根据输入维度生成一对可广播的 shape。
    较短 shape 的每个维度取 1 或对应较长 shape 的同维度值。
    每个维度的取值范围为 [0, 5]，确保 1 出现的概率足够高以覆盖广播分支。
    使用 testcase_name 派生确定性 seed，保证同一用例多次运行结果一致。
    """
    seed = zlib.crc32(testcase_name.encode()) if testcase_name else 0
    rng = np.random.default_rng(seed)

    if ndim1 < ndim2:
        small_ndim, large_ndim = ndim1, ndim2
        swapped = False
    else:
        small_ndim, large_ndim = ndim2, ndim1
        swapped = True

    large_shape = tuple(int(rng.integers(0, 6)) for _ in range(large_ndim))
    small_shape = tuple(
        int(rng.choice([1, large_shape[large_ndim - small_ndim + i]]))
        for i in range(small_ndim)
    )

    if swapped:
        return large_shape, small_shape
    else:
        return small_shape, large_shape


def broadcast_gradient_args_input(x1, x2, **kwargs):
    """
    Kernel input generator for broadcast_gradient_args.
    根据原始 x1/x2 的维度生成可广播的 shape 数据。
    维度取值范围为 [0, 5]，使用 testcase_name 派生确定性 seed。
    """
    print("ori_x1: ", x1)
    print("ori_x2: ", x2)
    x1_dim = x1.shape[0]
    x2_dim = x2.shape[0]
    testcase_name = kwargs.get("testcase_name", "")
    x1_tuple, x2_tuple = generate_broadcastable_shapes(x1_dim, x2_dim, testcase_name)
    x1_array = np.array(x1_tuple).astype(x1.dtype)
    x2_array = np.array(x2_tuple).astype(x2.dtype)

    print("x1: ", x1_array)
    print("x2: ", x2_array)
    return x1_array, x2_array
