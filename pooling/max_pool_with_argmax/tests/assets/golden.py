#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import numpy as np
from copy import deepcopy

__golden__ = {"kernel": {"max_pool_with_argmax": "max_pool_with_argmax_golden"}}


def nchw_to_nhwc_ksize_stride(v):
    """
    NCHW: [N, C, H, W]
    NHWC: [N, H, W, C]
    """
    return [v[0], v[2], v[3], v[1]]


def nhwc_index_to_nchw(idx, H, W, C):
    """
    Convert NHWC flatten index to NCHW flatten index
    """
    c = idx % C
    idx //= C
    w = idx % W
    idx //= W
    h = idx % H
    idx //= H
    n = idx
    return ((n * C + c) * H + h) * W + w


def max_pool_with_argmax_golden(
    x,
    *,
    ksize,
    strides,
    padding,
    targmax=9,
    include_batch_in_index=False,
    data_format="NHWC",
    nan_prop=False,
    **kwargs,
):
    """
    Golden function for max_pool_with_argmax.
    All the parameters (names and order) follow @max_pool_with_argmax_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    布局仅由 data_format 属性决定，与本仓 infershape/tiling 的属性单一来源实现一致；
    张量 format（input_formats/output_formats）仅为元数据，不参与布局计算。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensors (y, argmax)
    """
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1" if nan_prop else "0"

    import tensorflow as tf

    input_x = deepcopy(x)
    x_dtype = input_x.dtype

    ksize = list(ksize)
    strides = list(strides)
    include_batch = include_batch_in_index

    # ============================================================
    # 强制 NCHW → NHWC（golden 内部统一 NHWC）
    # 布局按 data_format 属性判定（与本仓 infershape/tiling 属性单一来源一致），
    # 张量 format 仅为元数据；属性 NCHW 时 kernel 按 NCHW 解释同一 buffer
    # ============================================================
    force_nhwc = False

    if data_format == "NCHW":
        force_nhwc = True

        input_x = input_x.transpose(0, 2, 3, 1)
        ksize = nchw_to_nhwc_ksize_stride(ksize)
        strides = nchw_to_nhwc_ksize_stride(strides)
        data_format = "NHWC"
    # 属性 NHWC：TF 与 kernel 对同一 buffer 的 NHWC 视图一致，直通

    x_tf = tf.convert_to_tensor(input_x)

    y_tf, arg_tf = tf.raw_ops.MaxPoolWithArgmax(
        input=x_tf,
        ksize=ksize,
        strides=strides,
        padding=padding,
        Targmax=tf.int64,
        include_batch_in_index=include_batch,
    )

    # ============================================================
    # 输出 dtype
    # ============================================================
    y_np = y_tf.numpy().astype(x_dtype)

    if targmax == 3:
        arg_np = arg_tf.numpy().astype(np.int32)
    else:
        arg_np = arg_tf.numpy().astype(np.int64)

    # ============================================================
    # NCHW index 语义修正
    # ============================================================
    if force_nhwc:
        N, H, W, C = input_x.shape
        arg_flat = arg_np.reshape(-1)
        for i in range(arg_flat.size):
            arg_flat[i] = nhwc_index_to_nchw(int(arg_flat[i]), H, W, C)
        arg_np = arg_flat.reshape(arg_np.shape)

    # ============================================================
    # 输出 format 转换：内部统一 NHWC，按 data_format 属性还原 NCHW 排布
    # （kernel 按属性布局写出 buffer，golden 与其对齐）
    # ============================================================
    if force_nhwc:
        y_np = y_np.transpose(0, 3, 1, 2)
        arg_np = arg_np.transpose(0, 3, 1, 2)

    return y_np, arg_np
