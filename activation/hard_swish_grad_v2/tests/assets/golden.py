#!/usr/bin/env python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import numpy as np
import torch

__spec__ = {
    "hard_swish_grad_v2": "HardSwishGradV2KernelSpec",
    "aclnnHardswishBackwardV2": "HardSwishGradV2AclnnSpec",
}

_TOL_KERNEL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}
_TOL_LOCAL = {
    "float32": {"standard": "stat_rel_err"},
    "float16": {"standard": "stat_rel_err"},
    "bfloat16": {"standard": "stat_rel_err"},
}


def _numpy_to_torch_tensor(array):
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _dtype_string(value):
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def _prepare_hard_swish_grad_v2_inputs(grad_output, self_x):
    source_tensor = (
        _numpy_to_torch_tensor(np.asarray(grad_output))
        if not isinstance(grad_output, torch.Tensor)
        else grad_output
    )
    compute_dtype = (
        torch.float32
        if source_tensor.dtype in (torch.float16, torch.bfloat16)
        else source_tensor.dtype
    )
    g = source_tensor.to(compute_dtype)
    x_source = (
        _numpy_to_torch_tensor(np.asarray(self_x))
        if not isinstance(self_x, torch.Tensor)
        else self_x
    )
    x = x_source.to(compute_dtype)
    return source_tensor, g, x, compute_dtype


def _cast_hard_swish_grad_v2_output(out, source_tensor, target=None):
    if target is None:
        target = (
            "bfloat16"
            if source_tensor.dtype == torch.bfloat16
            else str(source_tensor.dtype).replace("torch.", "")
        )
    if target == "bfloat16":
        return out.to(torch.bfloat16)
    return out.to(source_tensor.dtype)


def _hard_swish_grad_v2_compute(grad_output, self_x, target=None):
    source_tensor, g, x, compute_dtype = _prepare_hard_swish_grad_v2_inputs(
        grad_output, self_x
    )

    one_third = torch.tensor(0.33333334, dtype=compute_dtype, device=x.device)
    one_half = torch.tensor(0.5, dtype=compute_dtype, device=x.device)
    val = x * one_third + one_half

    mask_greater = x > torch.tensor(-3.0, dtype=compute_dtype, device=x.device)
    mask_less = x < torch.tensor(3.0, dtype=compute_dtype, device=x.device)
    val = torch.where(
        mask_greater, val, torch.tensor(0.0, dtype=compute_dtype, device=x.device)
    )
    val = torch.where(
        mask_less, val, torch.tensor(1.0, dtype=compute_dtype, device=x.device)
    )

    return _cast_hard_swish_grad_v2_output(g * val, source_tensor, target)


class _HardSwishGradV2Compose:
    def __call__(self, /, gradOutput, **kwargs):
        return [_hard_swish_grad_v2_compute(gradOutput, kwargs["self"])]


class HardSwishGradV2KernelSpec:
    @staticmethod
    def golden(gradOutput, self, **kwargs):
        output_dtypes = kwargs.get("output_dtypes")
        target = (
            _dtype_string(output_dtypes[0])
            if output_dtypes
            else str(np.asarray(gradOutput).dtype)
        )
        out = _hard_swish_grad_v2_compute(gradOutput, self, target)
        return [_torch_to_numpy_tensor(out)]

    third_party = {"torch": _HardSwishGradV2Compose}
    tolerance = _TOL_KERNEL


class HardSwishGradV2AclnnSpec:
    @staticmethod
    def golden(gradOutput, self, out=None, **kwargs):
        return [_hard_swish_grad_v2_compute(gradOutput, self)]

    third_party = {"torch": _HardSwishGradV2Compose}
    tolerance = _TOL_LOCAL


# Not registered in __spec__:
# - e2e/TensorFlow/ONNX/fusion: no torch_npu binding, parser, or graph pass is delivered.
