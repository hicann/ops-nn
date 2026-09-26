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

__golden__ = {
    "aclnn": {
        "aclnnNpuScatterAddBwd": "aclnn_npu_scatter_add_bwd_golden",
    },
    "kernel": {"npu_scatter_add_bwd": "npu_scatter_add_bwd_golden"},
}


def _scatter_add_bwd_reference(y_grad, x, s, indices):
    """按索引做fp32计算的反向基准，返回 (x_grad, s_grad) 的 fp32 结果。"""
    x_grad = np.zeros_like(x, dtype="float32")
    s_grad = np.zeros(s.shape, dtype="float32")
    y_grad_f = y_grad.astype("float32")
    x_f = x.astype("float32")
    s_f = s.astype("float32")
    for i in range(x.shape[0]):
        dst = int(indices[i])
        x_grad[i, :] = y_grad_f[dst, :] * s_f[i]
        s_grad[i] = np.sum(x_f[i, :] * y_grad_f[dst, :])
    return x_grad, s_grad


def npu_scatter_add_bwd_golden(y_grad, x, s, indices, **kwargs):
    """
    Golden function for npu_scatter_add_bwd.
    All the parameters (names and order) follow @npu_scatter_add_bwd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Returns:
        List of output tensors [x_grad, s_grad] (same dtype/shape as x, s)
    """
    x_grad, s_grad = _scatter_add_bwd_reference(y_grad, x, s, indices)
    return [x_grad.astype(x.dtype, copy=False), s_grad.astype(s.dtype, copy=False)]


def aclnn_npu_scatter_add_bwd_golden(y_grad, x, s, indices, x_grad, s_grad, **kwargs):
    """
    Aclnn golden for aclnnNpuScatterAddBwd.
    Parameters follow @aclnnNpuScatterAddBwdGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import torch

    x_grad_res = torch.zeros_like(x_grad).float()
    s_grad_res = torch.zeros_like(s_grad).float()
    y_grad_f = y_grad.float()
    x_f = x.float()
    s_f = s.float()
    for i in range(x.shape[0]):
        dst = int(indices[i].item())
        x_grad_res[i, :] = y_grad_f[dst, :] * s_f[i]
        s_grad_res[i] = (x_f[i, :] * y_grad_f[dst, :]).sum()
    return [x_grad_res.to(x_grad.dtype), s_grad_res.to(s_grad.dtype)]
