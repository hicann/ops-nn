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
    "inplace_apply_adagrad": "InplaceApplyAdagradKernelSpec",
    "InplaceApplyAdagrad": "InplaceApplyAdagradKernelSpec",
}

# Legacy TTK loaders inspect __golden__ / top-level *_golden functions.
__golden__ = {
    "kernel": {
        "inplace_apply_adagrad": "inplace_apply_adagrad_golden",
        "InplaceApplyAdagrad": "inplace_apply_adagrad_golden",
    },
}

_TOL = {
    # The default fp32 small-value absolute tolerance (2^-30) is below the arithmetic floor for
    # values produced by var - lr*grad/sqrt(accum): exact-zero accum and endpoint inputs can cancel
    # to ~1e-7. Use one fp32 ULP-scale absolute tolerance for the near-zero branch.
    "float32": {
        "standard": "cross_check",
        "level": "L1",
        "mare_ratio": 10.0,
        "small_value_atol": 1e-6,
    },
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _to_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    array = np.ascontiguousarray(value)
    if array.dtype.name == "bfloat16":
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _adagrad_compose(var, accum, lr, grad, update_slots):
    var_t = _to_torch(var)
    accum_t = _to_torch(accum)
    grad_t = _to_torch(grad)
    lr_t = _to_torch(lr).reshape(-1)[0]
    out_dtype = var_t.dtype
    compute_dtype = (
        torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
    )
    var_c = var_t.to(compute_dtype)
    accum_c = accum_t.to(compute_dtype)
    grad_c = grad_t.to(compute_dtype)
    lr_c = lr_t.to(compute_dtype)
    accum_out = accum_c + grad_c * grad_c if update_slots else accum_c
    var_out = var_c - lr_c * grad_c / torch.sqrt(accum_out)
    return [var_out.to(out_dtype), accum_out.to(out_dtype)]


def inplace_apply_adagrad_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    """Return the two outputs (var, accum) declared by the operator schema."""
    del use_locking, kwargs
    return _adagrad_compose(var, accum, lr, grad, update_slots)


class InplaceApplyAdagradTorchCompose:
    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        del use_locking, kwargs
        self.update_slots = update_slots

    def __call__(self, var, accum, lr, grad, **kwargs):
        del kwargs
        return _adagrad_compose(var, accum, lr, grad, self.update_slots)


class InplaceApplyAdagradKernelSpec:
    golden = staticmethod(inplace_apply_adagrad_golden)
    third_party = {"torch": InplaceApplyAdagradTorchCompose}
    tolerance = _TOL
