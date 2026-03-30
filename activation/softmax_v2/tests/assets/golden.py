#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS provided on an "AS IS" BASIS, WITHOUT warranties of any kind, either express or implied,
# including but not limited to non-INFRINGEMENT, merchantability or or fitness for a PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import torch
import copy

__golden__ = {
    "kernel": {
        "softmax_v2": "softmax_v2_golden"
    },
    "aclnn": {
        "aclnnSoftmax": "aclnn_softmax_golden"
    }
}


def _normalize_axis(axis, shape_length):
    normalized_axis = []
    if isinstance(axis, int):
        normalized_axis = [axis]
    elif isinstance(axis, tuple):
        normalized_axis = list(axis)
    elif isinstance(axis, list):
        normalized_axis = copy.deepcopy(axis)
    if not normalized_axis:
        normalized_axis = [-1]
    normalized_axis = [v if v >= 0 else v + shape_length for v in normalized_axis]
    normalized_axis = tuple(list(set(normalized_axis)))
    return normalized_axis


def _softmax(x, axis=None, output_dtype='float32'):
    x_dtype = x.dtype
    if "float16" in str(x_dtype):
        x = x.astype("float32", copy=True)
    reduce_max = np.amax(x, axis=axis, keepdims=True)
    sub_0 = np.subtract(x, reduce_max)
    exp_0 = np.exp(sub_0)
    reduce_sum = np.sum(exp_0, axis=axis, keepdims=True)
    out = np.divide(exp_0, reduce_sum)
    return out.astype(output_dtype, copy=False)


def softmax_v2_golden(x, *, axis=None, output_dtype='float32', **kwargs):
    input_ori_shapes = kwargs.get('input_ori_shapes', [[]])
    ori_shape = input_ori_shapes[0]
    
    if axis is None:
        axis = kwargs.get('axes')
    
    axis = _normalize_axis(axis, len(ori_shape))
    result = _softmax(x, axis, output_dtype)
    
    return result


def aclnn_softmax_golden(selfT, dim, out, **kwargs):
    '''
    Aclnn golden for aclnnSoftmax.
    All the parameters (name & order) follow \
        function `aclnnSoftmaxGetWorkspaceSize` in @aclnn_softmax.h \
        without `workspaceSize` & `executor`.
    When all dtypes are natively supported by torch, \
        the Tensors in the parameters are all torch.Tensor. \
        Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

    Args:
        kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

    Returns:
        Output tensors.
    '''
    input_data_dtype = selfT.dtype
    if "float16" in str(input_data_dtype):
        selfT = selfT.to(dtype=torch.float32)

    result = torch.softmax(selfT, dim=dim)
    result = result.to(dtype=out.dtype)
    return result

