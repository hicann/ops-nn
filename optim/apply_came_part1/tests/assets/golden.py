#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


__spec__ = {"apply_came_part1": "ApplyCamePart1KernelSpec"}


def _dtype_name(tensor):
    return getattr(
        np.asarray(tensor).dtype, "name", str(np.asarray(tensor).dtype)
    ).lower()


def _to_torch(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.contiguous()
    array = np.ascontiguousarray(tensor)
    if _dtype_name(array) == "bfloat16":
        return torch.from_numpy(array.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _compute(grad, eps, *, high_precision=False):
    grad_tensor = _to_torch(grad)
    if high_precision and grad_tensor.dtype.is_floating_point:
        grad_tensor = grad_tensor.to(torch.float64)
    elif grad_tensor.dtype in (torch.float16, torch.bfloat16):
        grad_tensor = grad_tensor.to(torch.float32)
    eps_scalar = _to_torch(eps).reshape(-1)[0].to(grad_tensor.dtype)
    values = torch.square(grad_tensor) + eps_scalar
    return [
        torch.sum(values, dim=-1),
        torch.sum(values, dim=-2),
        torch.sum(values, dim=(-2, -1)),
    ]


def _to_numpy(outputs):
    return tuple(output.detach().cpu().numpy() for output in outputs)


def apply_came_part1_golden(grad, eps, **kwargs):
    """Independent high-precision Torch reference for the three reductions."""
    return _to_numpy(_compute(grad, eps, high_precision=True))


class _ApplyCamePart1Compose:
    def __init__(self, **kwargs):
        pass

    def __call__(self, grad, eps, **kwargs):
        outputs = _compute(grad, eps)
        return [output.to(torch.float32) for output in outputs]


class ApplyCamePart1KernelSpec:
    golden = staticmethod(apply_came_part1_golden)
    third_party = {"torch": _ApplyCamePart1Compose}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }


# 【不存在】aclnn/e2e 通路：本算子在 ops-nn CMake 中标记 ACLNNTYPE aclnn_exclude；
# canndev 基线仅包含 GE/内核实现，未提供 ACLNN 或 torch_npu 接口。
