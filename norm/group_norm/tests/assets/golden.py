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


class GroupNormTestSpec:
    """GroupNorm CPU reference shared by Kernel and GEIR tests."""

    golden = staticmethod(group_norm_golden)


class TorchGroupNormTestSpec:
    """E2E golden for the public PyTorch GroupNorm API.

    ``torch.nn.functional.group_norm`` returns only ``y``.  The kernel/GEIR
    interface additionally exposes mean and variance, so those outputs are
    intentionally not fabricated in this E2E spec.
    """

    @staticmethod
    def golden(input, num_groups, weight=None, bias=None, eps=1e-5, **kwargs):
        del kwargs
        import torch.nn.functional as functional

        return [
            functional.group_norm(
                input,
                num_groups,
                weight=weight,
                bias=bias,
                eps=eps,
            )
        ]

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
    }
