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

import ml_dtypes
import numpy as np
import torch


__spec__ = {"inplace_apply_rms_prop": "InplaceApplyRMSPropKernelSpec"}
__golden__ = {"kernel": {"inplace_apply_rms_prop": "inplace_apply_rms_prop_golden"}}


_TOLERANCE = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _dtype_name(value):
    dtype = np.asarray(value).dtype
    return getattr(dtype, "name", str(dtype))


def _numpy_dtype(dtype_name):
    if dtype_name == "bfloat16":
        return ml_dtypes.bfloat16
    return np.dtype(dtype_name)


def _normalize_dtype_name(dtype):
    if isinstance(dtype, (list, tuple)):
        dtype = dtype[0] if dtype else None
    if dtype is None:
        return None
    name = str(dtype).lower().replace("torch.", "").replace("numpy.", "")
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float": "float32",
        "fp64": "float64",
        "double": "float64",
    }
    return aliases.get(name, name)


def _to_torch(value):
    array = np.asarray(value)
    if _dtype_name(array) == "bfloat16":
        return torch.from_numpy(np.asarray(array, dtype=np.float32))
    return torch.from_numpy(np.ascontiguousarray(array))


def inplace_apply_rms_prop_golden(
    var, ms, mom, lr, rho, momentum, epsilon, grad, output_dtype=None
):
    """PyTorch operator composition used by the NumPy-facing golden path."""
    source_dtype = _dtype_name(var)
    target_dtype_name = output_dtype or source_dtype
    target_dtype = _numpy_dtype(target_dtype_name)
    if np.asarray(var).size == 0:
        return [
            np.asarray(value).astype(target_dtype, copy=True)
            for value in (var, ms, mom)
        ]

    var_tensor, ms_tensor, mom_tensor, grad_tensor = map(
        _to_torch, (var, ms, mom, grad)
    )
    if var_tensor.dtype in (torch.float16, torch.bfloat16):
        var_tensor = var_tensor.float()
        ms_tensor = ms_tensor.float()
        mom_tensor = mom_tensor.float()
        grad_tensor = grad_tensor.float()

    lr_value = _to_torch(lr).reshape(-1)[0].to(var_tensor.dtype)
    rho_value = _to_torch(rho).reshape(-1)[0].to(var_tensor.dtype)
    momentum_value = _to_torch(momentum).reshape(-1)[0].to(var_tensor.dtype)
    epsilon_value = _to_torch(epsilon).reshape(-1)[0].to(var_tensor.dtype)
    epsilon_floor = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=var_tensor.dtype
    )
    epsilon_value = torch.where(
        epsilon_value > epsilon_floor, epsilon_value, epsilon_floor
    )

    # Keep the algebraic order used by the kernel. torch.lerp computes
    # grad^2 + rho * (ms - grad^2), which is mathematically equivalent but
    # can produce amplified var/mom differences after sqrt and division.
    ms_out = torch.add(
        torch.mul(ms_tensor, rho_value),
        torch.mul(torch.square(grad_tensor), 1.0 - rho_value),
    )
    denominator = torch.sqrt(torch.add(ms_out, epsilon_value))
    mom_out = torch.add(
        torch.mul(mom_tensor, momentum_value),
        torch.div(torch.mul(grad_tensor, lr_value), denominator),
    )
    var_out = torch.sub(var_tensor, mom_out)
    return [
        value.detach().cpu().numpy().astype(target_dtype, copy=False)
        for value in (var_out, ms_out, mom_out)
    ]


def _torch_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad):
    """Independent GPU composition used by the remote third-party provider."""
    target_dtype = var.dtype
    if target_dtype in (torch.float16, torch.bfloat16):
        var = var.float()
        ms = ms.float()
        mom = mom.float()
        grad = grad.float()

    lr_value = lr.reshape(-1)[0].to(var.dtype)
    rho_value = rho.reshape(-1)[0].to(var.dtype)
    momentum_value = momentum.reshape(-1)[0].to(var.dtype)
    epsilon_value = epsilon.reshape(-1)[0].to(var.dtype)
    epsilon_floor = torch.tensor(
        torch.finfo(torch.float32).tiny, dtype=var.dtype, device=var.device
    )
    epsilon_value = torch.where(
        epsilon_value > epsilon_floor, epsilon_value, epsilon_floor
    )

    ms_out = rho_value * ms + (1.0 - rho_value) * torch.square(grad)
    mom_out = momentum_value * mom + lr_value * grad / torch.sqrt(
        ms_out + epsilon_value
    )
    var_out = var - mom_out
    return [value.to(target_dtype) for value in (var_out, ms_out, mom_out)]


class _TorchRMSPropCompose:
    """Eager PyTorch composition used as the independent GPU baseline."""

    def __init__(self, use_locking=False, **kwargs):
        del use_locking, kwargs

    def __call__(self, var, ms, mom, lr, rho, momentum, epsilon, grad, **kwargs):
        del kwargs
        return _torch_rms_prop(var, ms, mom, lr, rho, momentum, epsilon, grad)


class InplaceApplyRMSPropKernelSpec:
    """Shared TestSpec for the kernel and GE IR invocation paths."""

    @staticmethod
    def golden(
        var,
        ms,
        mom,
        lr,
        rho,
        momentum,
        epsilon,
        grad,
        use_locking=False,
        **kwargs,
    ):
        output_dtypes = kwargs.get("output_dtypes") or []
        output_dtype = None
        if output_dtypes:
            output_dtype = _normalize_dtype_name(output_dtypes[0])
        del use_locking
        return inplace_apply_rms_prop_golden(
            var, ms, mom, lr, rho, momentum, epsilon, grad, output_dtype=output_dtype
        )

    third_party = {"torch": _TorchRMSPropCompose}
    tolerance = _TOLERANCE


# No aclnn or torch e2e TestSpec is registered because this operator only exposes
# the kernel and GE IR paths.
