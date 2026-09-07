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

"""Torch reference for Centralization kernel and GEIR validation."""

import numpy as np
import torch


def _normalized_axes(axes, rank):
    values = [-1] if axes is None or len(axes) == 0 else list(axes)
    return tuple(axis + rank if axis < 0 else axis for axis in values)


def _centralize(x, axes):
    compute = x.float() if x.dtype == torch.float16 else x
    result = compute - torch.mean(
        compute, dim=_normalized_axes(axes, x.dim()), keepdim=True
    )
    return result.to(x.dtype)


def _attr(kwargs, name, default):
    value = kwargs.get(name)
    if value is None and isinstance(kwargs.get("attributes"), dict):
        value = kwargs["attributes"].get(name)
    return default if value is None else value


def centralization_golden(x, **kwargs):
    axes = _attr(kwargs, "axes", [-1])
    value = torch.from_numpy(np.ascontiguousarray(x))
    return [_centralize(value, axes).numpy()]


class _CentralizationCompose:
    """Independent PyTorch implementation used by the remote GPU comparator."""

    def __init__(self, *, axes=(-1,), **_):
        self.axes = axes

    def __call__(self, x, **_):
        return [_centralize(x, self.axes)]


class CentralizationSpec:
    @staticmethod
    def golden(x, **kwargs):
        return centralization_golden(x, **kwargs)

    third_party = {"torch": _CentralizationCompose}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


__spec__ = {"centralization": "CentralizationSpec"}
__golden__ = {"kernel": {"centralization": "centralization_golden"}}
