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


__golden__ = {"kernel": {"add_rms_norm": "add_rms_norm_golden"}}

__spec__ = {
    # GEIR reuses the Kernel spec through the same op_name.
    "add_rms_norm": "AddRmsNormKernelSpec",
    "aclnnAddRmsNorm": "AclnnAddRmsNormSpec",
    # The repository has no public torch API; keep this key for downstream
    # wrappers that deliberately expose this API name.
    "torch.add_rms_norm": "TorchAddRmsNormSpec",
}


_TOLERANCE = {
    "float16": {"standard": "stat_rel_err", "threshold": 0.001},
    "bfloat16": {"standard": "stat_rel_err", "threshold": 0.008},
    "float32": {"standard": "stat_rel_err", "threshold": 0.001},
}


def add_rms_norm_golden(
    x1,
    x2,
    gamma,  # inputs
    epsilon: float = 1e-6,  # attributes
    **kwargs,
):
    """
    Golden function for add_rms_norm.
    All the parameters (names and order) follow @add_rms_norm_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    torch = importlib.import_module("torch")

    x1_dtype = x1.dtype
    if x1_dtype.name == "bfloat16":
        x1_tensor = torch.from_numpy(x1.view(np.float16)).view(torch.bfloat16)
        x2_tensor = torch.from_numpy(x2.view(np.float16)).view(torch.bfloat16)
        gamma_tensor = torch.from_numpy(gamma.view(np.float16)).view(torch.bfloat16)
    else:
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        gamma_tensor = torch.from_numpy(gamma)

    post_action = kwargs.get("_post_action")
    short_soc_version = kwargs.get("short_soc_version")
    if short_soc_version in ["Ascend910B", "Ascend910_93"]:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = add_rms_golden_v1(
            x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action
        )
    else:
        y_tensor, var_tensor, x_tensor, y_fp32_tensor = add_rms_golden_v2(
            x1_tensor, x2_tensor, gamma_tensor, epsilon, post_action
        )

    if x1_dtype.name == "bfloat16":
        y = y_tensor.view(torch.float16).numpy().view(x1_dtype)
        x = x_tensor.view(torch.float16).numpy().view(x1_dtype)
    else:
        y = y_tensor.numpy()
        x = x_tensor.numpy()
    rstd = var_tensor.numpy()
    if post_action == "cast":  # add_rms_norm_cast
        y_fp32 = y_fp32_tensor.numpy()
        return y_fp32, y, rstd, x
    else:
        return y, rstd, x


def add_rms_golden_v1(x1, x2, gamma, eps, post_action):
    torch = importlib.import_module("torch")

    # 不同分支走的cast方案不同
    if x1.dtype == torch.bfloat16:
        x = (x1.type(torch.float32) + x2.type(torch.float32)).type(x1.dtype)
    else:
        x = x1 + x2

    xFp32 = x.type(torch.float32)
    if 0 in xFp32.shape[: len(xFp32.shape) - len(gamma.shape)]:  # A 轴存在 0
        xFp32_1 = xFp32.reshape(
            (*xFp32.shape[: len(xFp32.shape) - len(gamma.shape)], 0)
        )
    else:
        xFp32_1 = xFp32.reshape(
            (*xFp32.shape[: len(xFp32.shape) - len(gamma.shape)], -1)
        )
    rstd = torch.rsqrt(
        xFp32_1.pow(2)
        .mean(-1, keepdim=True)
        .reshape(
            (
                *xFp32.shape[: len(xFp32.shape) - len(gamma.shape)],
                *([1] * len(gamma.shape)),
            )
        )
        + eps
    )
    tmpX = xFp32 * rstd

    if x1.dtype == torch.bfloat16:
        tmpX = tmpX.type(torch.bfloat16).type(torch.float32)
        y = (tmpX * gamma.type(torch.float32)).type(x1.dtype)
    elif x.dtype == torch.float16:
        tmpX = tmpX.type(torch.float16)
        y = tmpX * gamma
    else:
        y = tmpX * gamma
    y_fp32 = y.type(torch.float32) if post_action == "cast" else None
    return y, rstd, x, y_fp32


def add_rms_golden_v2(x1, x2, gamma, eps, post_action):
    torch = importlib.import_module("torch")

    x = x1.type(torch.float32) + x2.type(torch.float32)
    if 0 in x.shape[: len(x.shape) - len(gamma.shape)]:  # A 轴存在 0
        x_1 = x.reshape((*x.shape[: len(x.shape) - len(gamma.shape)], 0))
    else:
        x_1 = x.reshape((*x.shape[: len(x.shape) - len(gamma.shape)], -1))
    rstd = torch.rsqrt(
        x_1.pow(2)
        .mean(-1, keepdim=True)
        .reshape(
            (*x.shape[: len(x.shape) - len(gamma.shape)], *([1] * len(gamma.shape)))
        )
        + eps
    )
    tmpX = x * rstd

    y_fp32 = tmpX * gamma.type(torch.float32)
    y = y_fp32.type(x1.dtype)
    x = x.type(x1.dtype)
    return y, rstd, x, y_fp32


def _uses_legacy_math(short_soc_version):
    normalized = str(short_soc_version or "").lower().replace("_", "")
    return normalized in {"ascend910b", "ascend91093"}


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


def _numpy_golden(x1, x2, gamma, epsilon, legacy_math):
    input_dtype = x1.dtype
    dtype_name = input_dtype.name

    if legacy_math:
        if dtype_name == "bfloat16":
            x = (x1.astype(np.float32) + x2.astype(np.float32)).astype(input_dtype)
        else:
            x = x1 + x2
        x_fp32 = x.astype(np.float32)
        rstd = _numpy_rstd(x_fp32, gamma.ndim, epsilon)
        normalized = x_fp32 * rstd
        if dtype_name == "bfloat16":
            normalized = normalized.astype(input_dtype).astype(np.float32)
            y = (normalized * gamma.astype(np.float32)).astype(input_dtype)
        elif dtype_name == "float16":
            normalized = normalized.astype(input_dtype)
            y = (normalized * gamma).astype(input_dtype)
        else:
            y = normalized * gamma
        return [y, rstd, x.astype(input_dtype, copy=False)]

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


def _torch_golden(x1, x2, gamma, epsilon, legacy_math):
    # Dynamic import avoids eagerly initializing torch/torch_npu when TTK only
    # loads this module for a Kernel or GEIR TestSpec.
    torch = importlib.import_module("torch")
    input_dtype = x1.dtype

    if legacy_math:
        if input_dtype == torch.bfloat16:
            x = (x1.to(torch.float32) + x2.to(torch.float32)).to(input_dtype)
        else:
            x = x1 + x2
        x_fp32 = x.to(torch.float32)
        rstd = _torch_rstd(torch, x_fp32, gamma.dim(), epsilon)
        normalized = x_fp32 * rstd
        if input_dtype == torch.bfloat16:
            normalized = normalized.to(input_dtype).to(torch.float32)
            y = (normalized * gamma.to(torch.float32)).to(input_dtype)
        elif input_dtype == torch.float16:
            normalized = normalized.to(input_dtype)
            y = normalized * gamma
        else:
            y = normalized * gamma
        return [y, rstd, x.to(input_dtype)]

    x_fp32 = x1.to(torch.float32) + x2.to(torch.float32)
    rstd = _torch_rstd(torch, x_fp32, gamma.dim(), epsilon)
    y = (x_fp32 * rstd * gamma.to(torch.float32)).to(input_dtype)
    return [y, rstd, x_fp32.to(input_dtype)]


class AddRmsNormKernelSpec:
    """Kernel and GEIR golden; tensor inputs are numpy arrays."""

    @staticmethod
    def golden(x1, x2, gamma, epsilon=1e-6, **kwargs):
        return _numpy_golden(
            x1,
            x2,
            gamma,
            epsilon,
            _uses_legacy_math(kwargs.get("short_soc_version")),
        )

    tolerance = _TOLERANCE


class AclnnAddRmsNormSpec:
    """ACLNN golden; pure output tensors are intentionally not consumed."""

    @staticmethod
    def golden(
        x1,
        x2,
        gamma,
        epsilon=1e-6,
        yOut=None,
        rstdOut=None,
        xOut=None,
        **kwargs,
    ):
        del yOut, rstdOut, xOut
        return _torch_golden(
            x1,
            x2,
            gamma,
            epsilon,
            _uses_legacy_math(kwargs.get("short_soc_version")),
        )

    tolerance = _TOLERANCE


class TorchAddRmsNormSpec(AclnnAddRmsNormSpec):
    """Golden for a downstream ``torch.add_rms_norm`` wrapper."""
