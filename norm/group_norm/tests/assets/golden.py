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

# Kernel and GEIR both resolve the raw operator name.  GEIR intentionally
# reuses the Kernel spec, so one registration covers both paths.

__spec__ = {
    "group_norm": "GroupNormTestSpec",
    "torch.nn.functional.group_norm": "TorchGroupNormTestSpec",
}


def group_norm_golden(x, gamma, beta, *, num_groups, eps=1e-4, **kwargs):
    """910B-compatible GroupNorm reference using FP32 population statistics."""
    del kwargs
    output_dtype = x.dtype
    batch = x.shape[0]
    channel = x.shape[1]
    group_shape = (batch, num_groups)

    if x.size == 0:
        if batch != 0 or channel == 0:
            raise ValueError(
                "empty x is supported only when N is 0 and C is greater than 0"
            )
        y = np.empty_like(x)
        stats = np.empty(group_shape, dtype=output_dtype)
        return [y, stats, stats.copy()]

    x_fp32 = x.astype(np.float32)
    grouped = x_fp32.reshape(batch, num_groups, -1)
    mean_fp32 = np.mean(grouped, axis=2)
    variance_fp32 = np.mean(np.square(grouped - mean_fp32[..., None]), axis=2)
    rstd_fp32 = np.reciprocal(np.sqrt(variance_fp32 + np.float32(eps)))

    normalized = ((grouped - mean_fp32[..., None]) * rstd_fp32[..., None]).reshape(
        x.shape
    )
    broadcast_shape = (1, channel) + (1,) * (x.ndim - 2)
    y_fp32 = normalized * gamma.astype(np.float32).reshape(broadcast_shape)
    y_fp32 += beta.astype(np.float32).reshape(broadcast_shape)
    return [
        y_fp32.astype(output_dtype),
        mean_fp32.astype(output_dtype),
        variance_fp32.astype(output_dtype),
    ]


def aclnn_group_norm_golden(x, gamma, beta, *, num_groups, eps=1e-4, **kwargs):
    """ACLNN GroupNorm reference returning (y, mean, rstd) like aclnnGroupNorm."""
    del kwargs
    output_dtype = x.dtype
    batch = x.shape[0]
    channel = x.shape[1]
    group_shape = (batch, num_groups)

    if x.size == 0:
        # aclnn 空Tensor语义：meanOut 填充 0、rstdOut 填充 NaN（N=0 时输出为空）。
        if channel == 0:
            raise ValueError("empty x is supported only when C is greater than 0")
        y = np.empty_like(x)
        if batch == 0:
            mean = np.empty(group_shape, dtype=output_dtype)
            rstd = np.empty(group_shape, dtype=output_dtype)
        else:
            mean = np.zeros(group_shape, dtype=output_dtype)
            rstd = np.full(group_shape, np.nan, dtype=output_dtype)
        return [y, mean, rstd]

    x_fp32 = x.astype(np.float32)
    grouped = x_fp32.reshape(batch, num_groups, -1)
    mean_fp32 = np.mean(grouped, axis=2)
    variance_fp32 = np.mean(np.square(grouped - mean_fp32[..., None]), axis=2)
    rstd_fp32 = np.reciprocal(np.sqrt(variance_fp32 + np.float32(eps)))

    normalized = ((grouped - mean_fp32[..., None]) * rstd_fp32[..., None]).reshape(
        x.shape
    )
    broadcast_shape = (1, channel) + (1,) * (x.ndim - 2)
    y_fp32 = normalized * gamma.astype(np.float32).reshape(broadcast_shape)
    y_fp32 += beta.astype(np.float32).reshape(broadcast_shape)
    return [
        y_fp32.astype(output_dtype),
        mean_fp32.astype(output_dtype),
        rstd_fp32.astype(output_dtype),
    ]


def _torch_group_norm_third_party(x, gamma, beta, *, num_groups, eps=1e-4, **kwargs):
    """torch CPU third-party reference aligned with Kernel/GEIR outputs (y, mean, variance)."""
    del kwargs
    import torch

    x_t = torch.as_tensor(x)
    gamma_t = torch.as_tensor(gamma)
    beta_t = torch.as_tensor(beta)
    batch, channel = x_t.shape[0], x_t.shape[1]
    grouped = x_t.to(torch.float32).reshape(batch, num_groups, -1)
    mean = grouped.mean(dim=-1)
    variance = (grouped - mean[..., None]).pow(2).mean(dim=-1)
    rstd = torch.rsqrt(variance + eps)
    normalized = ((grouped - mean[..., None]) * rstd[..., None]).reshape(x_t.shape)
    broadcast_shape = (1, channel) + (1,) * (x_t.dim() - 2)
    y = normalized * gamma_t.to(torch.float32).reshape(broadcast_shape)
    y = y + beta_t.to(torch.float32).reshape(broadcast_shape)
    return [y.to(x_t.dtype), mean.to(x_t.dtype), variance.to(x_t.dtype)]


def _aten_group_norm_third_party(x, gamma, beta, *, num_groups, eps=1e-4, **kwargs):
    """torch aten third-party reference aligned with aclnnGroupNorm outputs (y, mean, rstd)."""
    del kwargs
    import torch

    x_t = torch.as_tensor(x).to(torch.float64)
    gamma_t = torch.as_tensor(gamma).to(torch.float64)
    beta_t = torch.as_tensor(beta).to(torch.float64)
    batch, channel = x_t.shape[0], x_t.shape[1]
    hxw = 1
    for dim in x_t.shape[2:]:
        hxw *= dim
    y, mean, rstd = torch.ops.aten.native_group_norm(
        x_t, gamma_t, beta_t, batch, channel, hxw, num_groups, eps
    )
    dtype = torch.as_tensor(x).dtype
    return [y.to(dtype), mean.to(dtype), rstd.to(dtype)]


class GroupNormTestSpec:
    """GroupNorm CPU reference shared by Kernel and GEIR tests."""

    golden = staticmethod(group_norm_golden)
    # 三方标杆：torch CPU 独立实现，输出对齐 Kernel/GEIR 的 (y, mean, variance)。
    third_party = {"torch": _torch_group_norm_third_party}


class AclnnGroupNormTestSpec:
    """ACLNN golden spec for aclnnGroupNorm (outputs: out, meanOut, rstdOut)."""

    golden = staticmethod(aclnn_group_norm_golden)
    # 三方标杆：torch aten.native_group_norm（与本仓 ATK executor 同源参考），
    # 输出对齐 aclnnGroupNorm 的 (out, meanOut, rstdOut)。
    third_party = {"torch": _aten_group_norm_third_party}

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }


class TorchGroupNormTestSpec:
    """E2E golden for the public PyTorch GroupNorm API.

    ``torch.nn.functional.group_norm`` returns only ``y``.  The kernel/GEIR
    interface additionally exposes mean and variance, so those outputs are
    intentionally not fabricated in this E2E spec.
    """

    @staticmethod
    def golden(input, num_groups, weight=None, bias=None, eps=1e-5, **kwargs):
        del kwargs
        import torch

        # 独立参考实现：FP32 总体统计量的手写计算，不调用被测公开通路
        # torch.nn.functional.group_norm，保证 E2E 参考独立性。
        batch, channel = input.shape[0], input.shape[1]
        grouped = input.to(torch.float32).reshape(batch, num_groups, -1)
        mean = grouped.mean(dim=-1, keepdim=True)
        variance = (grouped - mean).pow(2).mean(dim=-1, keepdim=True)
        normalized = (grouped - mean) * torch.rsqrt(variance + eps)
        y = normalized.reshape(input.shape)
        broadcast_shape = (1, channel) + (1,) * (input.dim() - 2)
        if weight is not None:
            y = y * weight.to(torch.float32).reshape(broadcast_shape)
        if bias is not None:
            y = y + bias.to(torch.float32).reshape(broadcast_shape)
        return [y.to(input.dtype)]

    # 三方标杆：公开 torch CPU 通路（cross_check 时的第三方对照实现）。
    third_party = {"torch": "torch.nn.functional.group_norm"}

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }
