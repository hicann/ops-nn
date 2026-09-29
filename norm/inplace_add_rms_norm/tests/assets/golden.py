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

import importlib

import numpy as np


__golden__ = {"kernel": {"inplace_add_rms_norm": "inplace_add_rms_norm_golden"}}

__spec__ = {
    # GEIR reuses the Kernel spec through the same op_name.
    "inplace_add_rms_norm": "InplaceAddRmsNormKernelSpec",
    "aclnnInplaceAddRmsNorm": "AclnnInplaceAddRmsNormSpec",
}


_TOLERANCE = {
    "float16": {"standard": "stat_rel_err", "threshold": 0.001},
    "bfloat16": {"standard": "stat_rel_err", "threshold": 0.008},
    "float32": {"standard": "stat_rel_err", "threshold": 0.001},
}


def inplace_add_rms_norm_golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
    """Generate y/rstd/x outputs for the Ascend 950 kernel ST."""
    torch = importlib.import_module("torch")

    del kwargs
    input_dtype = x1.dtype
    if input_dtype.name == "bfloat16":
        x1_tensor = torch.from_numpy(x1.view(np.float16)).view(torch.bfloat16)
        x2_tensor = torch.from_numpy(x2.view(np.float16)).view(torch.bfloat16)
        gamma_tensor = torch.from_numpy(gamma.view(np.float16)).view(torch.bfloat16)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        gamma_tensor = torch.from_numpy(gamma)

    x_fp32 = x1_tensor.float() + x2_tensor.float()
    norm_axis_begin = x_fp32.dim() - gamma_tensor.dim()
    reduce_shape = (*x_fp32.shape[:norm_axis_begin], -1)
    rstd_shape = (*x_fp32.shape[:norm_axis_begin], *([1] * gamma_tensor.dim()))
    rstd = torch.rsqrt(
        x_fp32.reshape(reduce_shape).square().mean(-1, keepdim=True) + epsilon
    )
    rstd = rstd.reshape(rstd_shape)
    y = (x_fp32 * rstd * gamma_tensor.float()).to(x1_tensor.dtype)
    x = x_fp32.to(x1_tensor.dtype)

    if input_dtype.name == "bfloat16":
        y_output = y.view(torch.float16).numpy().view(input_dtype)
        x_output = x.view(torch.float16).numpy().view(input_dtype)
    else:
        y_output = y.numpy()
        x_output = x.numpy()
    return y_output, rstd.numpy(), x_output


def _numpy_rstd(x_fp32, gamma_rank, epsilon):
    leading_shape = x_fp32.shape[:-gamma_rank] if gamma_rank else x_fp32.shape
    reduce_shape = x_fp32.shape[-gamma_rank:] if gamma_rank else ()
    reduce_size = int(np.prod(reduce_shape, dtype=np.int64)) if reduce_shape else 1
    if reduce_size == 0:
        mean_square = np.full((*leading_shape, 1), np.nan, dtype=np.float32)
    else:
        flattened = x_fp32.reshape((*leading_shape, reduce_size))
        mean_square = np.mean(flattened * flattened, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        rstd = np.reciprocal(np.sqrt(mean_square + np.float32(epsilon)))
    return rstd.reshape((*leading_shape, *([1] * gamma_rank))).astype(
        np.float32, copy=False
    )


def _numpy_golden(x1, x2, gamma, epsilon):
    input_dtype = x1.dtype
    x_fp32 = x1.astype(np.float32) + x2.astype(np.float32)
    rstd = _numpy_rstd(x_fp32, gamma.ndim, epsilon)
    y = (x_fp32 * rstd * gamma.astype(np.float32)).astype(input_dtype)
    return [y, rstd, x_fp32.astype(input_dtype)]


def _torch_rstd(torch, x_fp32, gamma_rank, epsilon):
    leading_shape = (
        tuple(x_fp32.shape[:-gamma_rank]) if gamma_rank else tuple(x_fp32.shape)
    )
    reduce_shape = tuple(x_fp32.shape[-gamma_rank:]) if gamma_rank else ()
    reduce_size = 1
    for dimension in reduce_shape:
        reduce_size *= int(dimension)
    if reduce_size == 0:
        mean_square = torch.full(
            (*leading_shape, 1),
            float("nan"),
            dtype=torch.float32,
            device=x_fp32.device,
        )
    else:
        flattened = x_fp32.reshape((*leading_shape, reduce_size))
        mean_square = flattened.square().mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(mean_square + float(epsilon))
    return rstd.reshape((*leading_shape, *([1] * gamma_rank)))


def _torch_golden(x1, x2, gamma, epsilon):
    # Dynamic import avoids eagerly initializing torch/torch_npu when TTK only
    # loads this module for a Kernel or GEIR TestSpec.
    torch = importlib.import_module("torch")
    input_dtype = x1.dtype

    x_fp32 = x1.to(torch.float32) + x2.to(torch.float32)
    rstd = _torch_rstd(torch, x_fp32, gamma.dim(), epsilon)
    y = (x_fp32 * rstd * gamma.to(torch.float32)).to(input_dtype)
    return [y, rstd, x_fp32.to(input_dtype)]


class InplaceAddRmsNormKernelSpec:
    """Kernel and GEIR golden; outputs map to x1, rstd and x2 aliases."""

    @staticmethod
    def golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
        del kwargs
        return _numpy_golden(x1, x2, gamma, epsilon)

    tolerance = _TOLERANCE


class AclnnInplaceAddRmsNormSpec:
    """ACLNN golden for x1Ref/x2Ref mutation and the rstd output."""

    @staticmethod
    def golden(x1Ref, x2Ref, gamma, epsilon=1e-6, rstdOut=None, **kwargs):
        del rstdOut, kwargs
        return _torch_golden(x1Ref, x2Ref, gamma, epsilon)

    tolerance = _TOLERANCE
