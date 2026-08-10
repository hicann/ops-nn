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
import torch


__spec__ = {
    "apply_adagrad": "ApplyAdagradKernelSpec",
}
__golden__ = {"kernel": {"apply_adagrad": "apply_adagrad_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _apply_adagrad_compute(var, accum, lr, grad, update_slots=True):
    input_dtype = (
        np.asarray(var).dtype if not isinstance(var, torch.Tensor) else var.dtype
    )
    dtype_name = getattr(input_dtype, "name", str(input_dtype))
    if isinstance(var, torch.Tensor):
        target_dtype = var.dtype
        var_tensor = var.clone()
        accum_tensor = accum.clone()
        grad_tensor = grad.clone()
        if var_tensor.dtype in (torch.float16, torch.bfloat16):
            var_tensor = var_tensor.to(torch.float32)
            accum_tensor = accum_tensor.to(torch.float32)
            grad_tensor = grad_tensor.to(torch.float32)
        lr_value = float(lr.reshape(-1)[0].detach().cpu())
    else:
        compute_dtype = (
            torch.float32
            if dtype_name in ("float16", "bfloat16")
            else torch.from_numpy(np.asarray(var)).dtype
        )
        var_array = var.astype(np.float32) if dtype_name == "bfloat16" else var
        accum_array = accum.astype(np.float32) if dtype_name == "bfloat16" else accum
        grad_array = grad.astype(np.float32) if dtype_name == "bfloat16" else grad
        var_tensor = torch.tensor(var_array, dtype=compute_dtype)
        accum_tensor = torch.tensor(accum_array, dtype=compute_dtype)
        grad_tensor = torch.tensor(grad_array, dtype=compute_dtype)
        lr_value = float(np.asarray(lr).reshape(-1)[0])

    if update_slots:
        step_tensor = torch.tensor(1.0)
        torch.optim._functional.adagrad(
            [var_tensor],
            [grad_tensor],
            [accum_tensor],
            [step_tensor],
            fused=False,
            grad_scale=None,
            found_inf=None,
            has_sparse_grad=False,
            foreach=False,
            differentiable=False,
            has_complex=False,
            lr=lr_value,
            weight_decay=0.0,
            lr_decay=0.0,
            eps=0.0,
            maximize=False,
        )
    else:
        var_tensor = var_tensor - lr_value * grad_tensor / torch.sqrt(accum_tensor)
    if isinstance(var, torch.Tensor):
        var_tensor = var_tensor.to(target_dtype)
        accum_tensor = accum_tensor.to(target_dtype)
    return [var_tensor, accum_tensor]


class _ApplyAdagradCompose:
    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        self.update_slots = update_slots

    def __call__(self, var, accum, lr, grad, **kwargs):
        return _apply_adagrad_compute(var, accum, lr, grad, self.update_slots)


class ApplyAdagradKernelSpec:
    @staticmethod
    def golden(var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs):
        outs = _apply_adagrad_compute(var, accum, lr, grad, update_slots)
        input_dtype = np.asarray(var).dtype
        return [out.numpy().astype(input_dtype, copy=False) for out in outs]

    third_party = {"torch": _ApplyAdagradCompose}
    tolerance = _TOL


def apply_adagrad_golden(
    var,
    accum,
    lr,
    grad,  # inputs
    update_slots: bool = True,
    use_locking: bool = False,  # attributes
    **kwargs,
):
    """
    Golden function for apply_adagrad.
    All the parameters (names and order) follow @apply_adagrad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor and mutable accum tensor
    """
    input_dtype = var.dtype
    res_var, res_accum = _apply_adagrad_compute(var, accum, lr, grad, update_slots)
    return [
        res_var.numpy().astype(input_dtype, copy=False),
        res_accum.numpy().astype(input_dtype, copy=False),
    ]


# Not registered in __spec__:
# - aclnn/e2e/ONNX/fusion: ApplyAdagrad is aclnn_exclude and has no torch/ONNX/fusion path.
# - TensorFlow parser is validated as a framework parser route, not as a TestSpec api_name.
