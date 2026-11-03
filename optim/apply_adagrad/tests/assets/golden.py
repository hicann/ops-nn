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

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

__spec__ = {
    "apply_adagrad": "ApplyAdagradKernelSpec",
    "ApplyAdagrad": "ApplyAdagradKernelSpec",
    "tf.raw_ops.ApplyAdagrad": "ApplyAdagradTfRefSpec",
    "tf.raw_ops.ResourceApplyAdagrad": "ApplyAdagradTfResourceSpec",
}

# Keep the legacy TTK loader compatible with the multi-path Spec.  Newer
# runners resolve ``__spec__``; older codecheck probes only inspect
# ``__golden__`` or a top-level ``*_golden`` function.
__golden__ = {
    "kernel": {
        "apply_adagrad": "apply_adagrad_golden",
        "ApplyAdagrad": "apply_adagrad_golden",
    },
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
_GOLDEN_PYTHON_ENV = "APPLY_ADAGRAD_GOLDEN_PYTHON"


def _run_tf_apply_adagrad(
    var,
    accum,
    lr,
    grad,
    update_slots=True,
    use_locking=False,
    compute_dtype=None,
    **kwargs,
):
    del kwargs
    if isinstance(var, torch.Tensor):
        native_dtype = (
            "bfloat16"
            if var.dtype == torch.bfloat16
            else str(var.detach().cpu().numpy().dtype)
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
        native_dtype = var_array.dtype.name
    if native_dtype == "bfloat16":
        var_array = np.asarray(var_array, dtype=np.float32)
        accum_array = np.asarray(accum_array, dtype=np.float32)
        grad_array = np.asarray(grad_array, dtype=np.float32)
        lr_value = np.asarray(lr_value, dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="apply_adagrad_golden_") as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "output.npz"
        np.savez(
            input_path,
            var=np.ascontiguousarray(var_array),
            accum=np.ascontiguousarray(accum_array),
            lr=np.asarray(lr_value),
            grad=np.ascontiguousarray(grad_array),
            update_slots=np.asarray(update_slots, dtype=np.bool_),
            use_locking=np.asarray(use_locking, dtype=np.bool_),
            native_dtype=np.asarray(native_dtype),
            compute_dtype=np.asarray(compute_dtype or ""),
        )
        worker_env = os.environ.copy()
        worker_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        worker_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        worker_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        worker_env.pop("LD_LIBRARY_PATH", None)
        python_executable = worker_env.get(_GOLDEN_PYTHON_ENV, sys.executable)
        subprocess.run(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--tensorflow-worker",
                str(input_path),
                str(output_path),
            ],
            check=True,
            env=worker_env,
        )
        with np.load(output_path) as output_data:
            var_out = output_data["var"].copy()
            accum_out = output_data["accum"].copy()
        if native_dtype == "bfloat16":
            from ml_dtypes import bfloat16

            var_out = var_out.astype(bfloat16)
            accum_out = accum_out.astype(bfloat16)
        return [var_out, accum_out]


def _run_tensorflow_worker(input_path, output_path):
    with np.load(input_path) as input_data:
        var_array = np.ascontiguousarray(input_data["var"])
        accum_array = np.ascontiguousarray(input_data["accum"])
        grad_array = np.ascontiguousarray(input_data["grad"])
        lr_value = np.asarray(input_data["lr"]).reshape(-1)[0]
        update_slots = bool(input_data["update_slots"].item())
        use_locking = bool(input_data["use_locking"].item())
        native_dtype_name = str(input_data["native_dtype"].item())
        compute_dtype_name = str(input_data["compute_dtype"].item())

    import tensorflow as tf

    native_dtype = tf.as_dtype(native_dtype_name)
    tf_dtype = tf.as_dtype(compute_dtype_name) if compute_dtype_name else native_dtype
    var_array = np.asarray(var_array, dtype=tf_dtype.as_numpy_dtype)
    accum_array = np.asarray(accum_array, dtype=tf_dtype.as_numpy_dtype)
    grad_array = np.asarray(grad_array, dtype=tf_dtype.as_numpy_dtype)
    lr_value = np.asarray(lr_value, dtype=tf_dtype.as_numpy_dtype)

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
    if native_dtype != tf_dtype:
        with np.errstate(over="ignore", invalid="ignore"):
            var_out = np.asarray(var_out, dtype=native_dtype.as_numpy_dtype)
            accum_out = np.asarray(accum_out, dtype=native_dtype.as_numpy_dtype)
    if native_dtype == tf.bfloat16:
        var_out = np.asarray(var_out, dtype=np.float32)
        accum_out = np.asarray(accum_out, dtype=np.float32)
    np.savez(output_path, var=np.asarray(var_out), accum=np.asarray(accum_out))


def apply_adagrad_golden(
    var, accum, lr, grad, update_slots=True, use_locking=False, **kwargs
):
    """Return the single output declared by the ApplyAdagrad kernel schema."""
    del use_locking, kwargs
    output = _ApplyAdagradKernelTorchCompose(update_slots=update_slots)(
        var, accum, lr, grad
    )[0]
    if isinstance(var, torch.Tensor):
        return [output]

    output_dtype = np.asarray(var).dtype
    if output.dtype == torch.bfloat16:
        output = output.to(torch.float32)
    return [output.detach().cpu().numpy().astype(output_dtype, copy=False)]


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


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--tensorflow-worker":
        raise SystemExit(
            "golden.py is a TTK plugin; direct execution requires --tensorflow-worker"
        )
    _run_tensorflow_worker(sys.argv[2], sys.argv[3])
