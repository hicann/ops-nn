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

from typing import ClassVar

import numpy as np
import torch

__spec__ = {
    "apply_adagrad": "ApplyAdagradKernelSpec",
    "tf.raw_ops.ApplyAdagrad": "ApplyAdagradTfRefSpec",
    "tf.raw_ops.ResourceApplyAdagrad": "ApplyAdagradTfResourceSpec",
}

# Keep the legacy TTK loader compatible with the multi-path Spec.  Newer
# runners resolve ``__spec__``; older codecheck probes only inspect
# ``__golden__`` or a top-level ``*_golden`` function.
__golden__ = {
    "kernel": {"apply_adagrad": "apply_adagrad_golden"},
    "e2e": {
        "tf.raw_ops.ApplyAdagrad": "apply_adagrad_tf_ref_golden",
        "tf.raw_ops.ResourceApplyAdagrad": "apply_adagrad_tf_resource_golden",
    },
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
}


def _run_tf_apply_adagrad(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    import tensorflow as tf

    del kwargs
    # TTK normally supplies numpy arrays (including ml-dtypes bfloat16), while
    # direct smoke callers may supply torch tensors.  CPU torch cannot expose
    # bfloat16 through numpy, so promote only the transport representation and
    # keep the TensorFlow reference dtype explicit.
    if isinstance(var, torch.Tensor):
        tf_dtype = (
            tf.bfloat16
            if var.dtype == torch.bfloat16
            else tf.as_dtype(var.detach().cpu().numpy().dtype)
        )
        var_array = (
            var.detach().cpu().float().numpy()
            if var.dtype == torch.bfloat16
            else var.detach().cpu().numpy()
        )
        accum_array = (
            accum.detach().cpu().float().numpy()
            if accum.dtype == torch.bfloat16
            else accum.detach().cpu().numpy()
        )
        grad_array = (
            grad.detach().cpu().float().numpy()
            if grad.dtype == torch.bfloat16
            else grad.detach().cpu().numpy()
        )
        lr_value = (
            lr.detach().cpu().float().numpy()
            if lr.dtype == torch.bfloat16
            else lr.detach().cpu().numpy()
        ).reshape(-1)[0]
    else:
        var_array = np.asarray(var)
        accum_array = np.asarray(accum)
        grad_array = np.asarray(grad)
        lr_value = np.asarray(lr).reshape(-1)[0]
        tf_dtype = tf.as_dtype(var_array.dtype)

    graph = tf.Graph()
    with graph.as_default():
        var_ref = tf.compat.v1.Variable(var_array, dtype=tf_dtype, use_resource=False)
        accum_ref = tf.compat.v1.Variable(
            accum_array, dtype=tf_dtype, use_resource=False
        )
        output = tf.raw_ops.ApplyAdagrad(
            var=var_ref,
            accum=accum_ref,
            lr=tf.constant(lr_value, dtype=tf_dtype),
            grad=tf.constant(grad_array, dtype=tf_dtype),
            update_slots=update_slots,
            use_locking=use_locking,
        )
        init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as session:
        session.run(init)
        var_out, accum_out = session.run([output, accum_ref])
    return [np.asarray(var_out), np.asarray(accum_out)]


def apply_adagrad_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    """Return the single output declared by the ApplyAdagrad kernel schema."""
    return _run_tf_apply_adagrad(
        var, accum, lr, grad, update_slots, use_locking, **kwargs
    )[:1]


class _ApplyAdagradTfCompose:
    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        self.update_slots = update_slots
        self.use_locking = use_locking

    def __call__(self, var, accum, lr, grad, **kwargs):
        import tensorflow as tf

        del kwargs
        var_ref = tf.Variable(tf.convert_to_tensor(var), trainable=False)
        accum_ref = tf.Variable(tf.convert_to_tensor(accum), trainable=False)
        grad_tensor = tf.convert_to_tensor(grad, dtype=var_ref.dtype)
        lr_scalar = tf.reshape(tf.convert_to_tensor(lr, dtype=var_ref.dtype), [-1])[0]
        tf.raw_ops.ResourceApplyAdagrad(
            var=var_ref.handle,
            accum=accum_ref.handle,
            lr=lr_scalar,
            grad=grad_tensor,
            update_slots=self.update_slots,
            use_locking=self.use_locking,
        )
        return [var_ref.read_value(), accum_ref.read_value()]


def _to_torch_adagrad(value):
    if isinstance(value, torch.Tensor):
        return value
    array = np.ascontiguousarray(value)
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


class _ApplyAdagradTorchCompose:
    """Independent Torch composition for the kernel provider."""

    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        del use_locking, kwargs
        self.update_slots = update_slots

    def __call__(self, var, accum, lr, grad, **kwargs):
        del kwargs
        var_t = _to_torch_adagrad(var)
        accum_t = _to_torch_adagrad(accum)
        grad_t = _to_torch_adagrad(grad)
        lr_t = _to_torch_adagrad(lr).reshape(-1)[0]
        out_dtype = var_t.dtype
        compute_dtype = (
            torch.float32 if out_dtype in (torch.float16, torch.bfloat16) else out_dtype
        )
        var_c = var_t.to(compute_dtype)
        accum_c = accum_t.to(compute_dtype)
        grad_c = grad_t.to(compute_dtype)
        lr_c = lr_t.to(compute_dtype)
        accum_out = accum_c + grad_c * grad_c if self.update_slots else accum_c
        var_out = var_c - lr_c * grad_c / torch.sqrt(accum_out)
        return [var_out.to(out_dtype), accum_out.to(out_dtype)]


class _ApplyAdagradKernelTorchCompose(_ApplyAdagradTorchCompose):
    """Kernel provider exposes only the var output declared by OpDef."""

    def __call__(self, var, accum, lr, grad, **kwargs):
        return super().__call__(var, accum, lr, grad, **kwargs)[:1]


class _ApplyAdagradTfRefCompose:
    def __init__(self, update_slots=True, use_locking=False, **kwargs):
        del kwargs
        self.update_slots = update_slots
        self.use_locking = use_locking

    def __call__(self, var, accum, lr, grad, **kwargs):
        del kwargs
        return _run_tf_apply_adagrad(
            var,
            accum,
            lr,
            grad,
            update_slots=self.update_slots,
            use_locking=self.use_locking,
        )


class ApplyAdagradKernelSpec:
    golden = staticmethod(apply_adagrad_golden)

    third_party: ClassVar[dict] = {
        "torch": _ApplyAdagradKernelTorchCompose,
    }
    tolerance = _TOL


def _apply_adagrad_tf_api_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    return _run_tf_apply_adagrad(
        var, accum, lr, grad, update_slots, use_locking, **kwargs
    )


def apply_adagrad_tf_ref_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    return _apply_adagrad_tf_api_golden(
        var, accum, lr, grad, update_slots, use_locking, **kwargs
    )


def apply_adagrad_tf_resource_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    return _apply_adagrad_tf_api_golden(
        var, accum, lr, grad, update_slots, use_locking, **kwargs
    )


class ApplyAdagradTfRefSpec:
    golden = staticmethod(apply_adagrad_tf_ref_golden)
    third_party: ClassVar[dict] = {"torch": _ApplyAdagradTorchCompose}
    tolerance = _TOL


class ApplyAdagradTfResourceSpec:
    golden = staticmethod(apply_adagrad_tf_resource_golden)
    third_party: ClassVar[dict] = {"torch": _ApplyAdagradTorchCompose}
    tolerance = _TOL
