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

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _prepare_apply_adagrad_inputs(var, accum, lr, grad):
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
        target_dtype = None
    return var_tensor, accum_tensor, grad_tensor, lr_value, target_dtype


def _cast_apply_adagrad_outputs(var_tensor, accum_tensor, target_dtype):
    if target_dtype is not None:
        var_tensor = var_tensor.to(target_dtype)
        accum_tensor = accum_tensor.to(target_dtype)
    return [var_tensor, accum_tensor]


def _apply_adagrad_golden_compute(var, accum, lr, grad, update_slots=True):
    var_tensor, accum_tensor, grad_tensor, lr_value, target_dtype = (
        _prepare_apply_adagrad_inputs(var, accum, lr, grad)
    )
    accum_out = (
        accum_tensor + grad_tensor * grad_tensor if update_slots else accum_tensor
    )
    var_out = var_tensor - lr_value * grad_tensor / torch.sqrt(accum_out)
    return _cast_apply_adagrad_outputs(var_out, accum_out, target_dtype)


def _apply_adagrad_third_party_compute(var, accum, lr, grad, update_slots=True):
    var_tensor, accum_tensor, grad_tensor, lr_value, target_dtype = (
        _prepare_apply_adagrad_inputs(var, accum, lr, grad)
    )
    grad_square = grad_tensor * grad_tensor
    if update_slots:
        accum_tensor = accum_tensor + grad_square
    else:
        accum_tensor = accum_tensor + torch.zeros_like(grad_square)
    denom = torch.sqrt(accum_tensor)
    scaled_grad = grad_tensor * lr_value
    scaled_grad = scaled_grad / denom
    var_tensor = var_tensor - scaled_grad
    return _cast_apply_adagrad_outputs(var_tensor, accum_tensor, target_dtype)


class _ApplyAdagradCompose:
    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        self.update_slots = update_slots

    def __call__(self, var, accum, lr, grad, **kwargs):
        return _apply_adagrad_third_party_compute(
            var, accum, lr, grad, self.update_slots
        )


class ApplyAdagradKernelSpec:
    @staticmethod
    def golden(var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs):
        outs = _apply_adagrad_golden_compute(var, accum, lr, grad, update_slots)
        input_dtype = np.asarray(var).dtype
        return [out.numpy().astype(input_dtype, copy=False) for out in outs]

    third_party = {"torch": _ApplyAdagradCompose}
    tolerance = _TOL


# Not registered in __spec__:
# - aclnn/e2e/ONNX/fusion: ApplyAdagrad is aclnn_exclude and has no torch/ONNX/fusion path.
# - TensorFlow parser is validated as a framework parser route, not as a TestSpec api_name.
