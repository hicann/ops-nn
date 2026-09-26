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
        "aclnnNpuScatterAdd": "aclnn_npu_scatter_add_golden",
    },
    "kernel": {"npu_scatter_add": "npu_scatter_add_golden"},
}


def _to_float32(tensor):
    return tensor.astype("float32")


def _scatter_add_reference(x, y, s, indices, valid_rows):
    """按索引做fp32累加的高精度基准，返回fp32结果。"""
    result = _to_float32(y)
    x_f = _to_float32(x)
    for i in range(valid_rows):
        dst = int(indices[i])
        if s is None:
            result[dst, :] += x_f[i, :]
        else:
            result[dst, :] += x_f[i, :] * float(s[i])
    return result


def npu_scatter_add_golden(
    x, y, s, indices, sort_idx, valid_token_num, *, use_high_precision=False, **kwargs
):
    """
    Golden function for npu_scatter_add.
    All the parameters (names and order) follow @npu_scatter_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (same shape/dtype as y)
    """
    y_dtype = y.dtype
    valid_rows = x.shape[0] if valid_token_num is None else int(valid_token_num)
    result = _scatter_add_reference(x, y, s, indices, valid_rows)
    return result.astype(y_dtype, copy=False)


def aclnn_npu_scatter_add_golden(
    x, y, s, indices, sortIdx, validTokenNum, useHighPrecision, **kwargs
):
    """
    Aclnn golden for aclnnNpuScatterAdd.
    Parameters follow @aclnnNpuScatterAddGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    if s is None:
        s = torch.ones_like(indices, dtype=x.dtype)
    valid_rows = x.shape[0] if validTokenNum is None else int(validTokenNum.item())

    result = y.clone().float()
    x_f = x.float()
    s_f = s.float()
    for i in range(valid_rows):
        dst = int(indices[i].item())
        result[dst, :] += x_f[i, :] * s_f[i]
    return result.to(y.dtype)
