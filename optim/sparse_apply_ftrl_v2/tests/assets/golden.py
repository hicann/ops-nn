# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import hashlib
from typing import ClassVar

import numpy as np

__golden__ = {"kernel": {"sparse_apply_ftrl_v2": "sparse_apply_ftrl_v2_golden"}}

__input__ = {"kernel": {"sparse_apply_ftrl_v2": "sparse_apply_ftrl_v2_input"}}

__spec__ = {"sparse_apply_ftrl_v2": "SparseApplyFtrlV2TestSpec"}


def _magnitude_bound(dtype):
    """Safe magnitude bound for var/linear/grad so intermediate quantities stay finite.

    Keeps grad**2 and accum within the destination dtype's finite range, including
    the worst case where several indices target the same row (M > N).
    """
    try:
        name = np.dtype(dtype).name
        kind = np.dtype(dtype).kind
        itemsize = np.dtype(dtype).itemsize
    except TypeError:
        name = str(dtype)
        kind = ""
        itemsize = 0
    if name == "float16" or name == "bfloat16":
        return 10.0
    if kind == "V" and itemsize == 2:  # numpy-visible bfloat16 (void form)
        return 10.0
    return 1.0e3


def _stable_seed(name):
    """Deterministic 64-bit seed derived from the testcase name (0 when absent)."""
    text = str(name) if name is not None else ""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _finite_clip(arr, lo, hi):
    """Replace non-finite values and clip into [lo, hi] in fp32, preserving dtype."""
    out_dtype = np.asarray(arr).dtype
    values = np.asarray(arr).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=float(hi), neginf=float(lo))
    values = np.clip(values, lo, hi)
    return values.astype(out_dtype)


def sparse_apply_ftrl_v2_input(
    var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power, **kwargs
):
    """
    Custom input generator: constrain inputs to the documented valid domain.

    The README only accepts finite values in a numerically safe range; outside it
    intermediate terms (grad**2, accum**(-lr_power)) overflow and the result is
    implementation-specific rather than a defect. Constraints applied here:

      - var/linear/grad: finite and bounded per dtype (see _magnitude_bound)
      - accum: finite and in [1e-3, 100] (squared-gradient accumulator)
      - lr:   finite and in [1e-3, 1]
      - l1/l2/l2_shrinkage: finite and in [0, 1]
      - lr_power: finite and in [-1, -1e-3] (non-positive)
      - indices: valid, in [0, N); unique when M <= N, deterministic in-range
        pattern when M > N (duplicates are still applied in ascending index
        order, matching TensorFlow)
    """
    mag = _magnitude_bound(var.dtype)

    var = _finite_clip(var, -mag, mag)
    linear = _finite_clip(linear, -mag, mag)
    grad = _finite_clip(grad, -mag, mag)
    accum = _finite_clip(accum, 1e-3, 100.0)
    lr = _finite_clip(lr, 1e-3, 1.0)
    l1 = _finite_clip(l1, 0.0, 1.0)
    l2 = _finite_clip(l2, 0.0, 1.0)
    l2_shrinkage = _finite_clip(l2_shrinkage, 0.0, 1.0)
    lr_power = _finite_clip(lr_power, -1.0, -1e-3)

    N = var.shape[0]
    M = indices.shape[0]
    if M > 0 and N > 0:
        # Deterministic per testcase so repeated generations of the same case
        # produce identical indices (reproducible inputs across modes/routes).
        seed = _stable_seed(kwargs.get("testcase_name"))
        rng = np.random.default_rng(seed)
        if M <= N:
            valid_indices = rng.permutation(N)[:M]
        else:
            valid_indices = np.arange(M) % N
        indices = valid_indices.astype(indices.dtype)

    return [var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power]


def _run_tf_ftrl_v2(
    var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power, use_locking
):
    """Run tf.raw_ops.ResourceSparseApplyFtrlV2 on already-prepared arrays."""
    import tensorflow as tf

    def to_idx(arr):
        return tf.constant(arr, dtype=tf.int32 if arr.dtype == np.int32 else tf.int64)

    v_var = tf.Variable(var, name="var")
    v_accum = tf.Variable(accum, name="accum")
    v_linear = tf.Variable(linear, name="linear")

    lr_val = lr.item() if hasattr(lr, "item") else float(lr)
    l1_val = l1.item() if hasattr(l1, "item") else float(l1)
    l2_val = l2.item() if hasattr(l2, "item") else float(l2)
    l2s_val = (
        l2_shrinkage.item() if hasattr(l2_shrinkage, "item") else float(l2_shrinkage)
    )
    lrp_val = lr_power.item() if hasattr(lr_power, "item") else float(lr_power)

    tf.raw_ops.ResourceSparseApplyFtrlV2(
        var=v_var.handle,
        accum=v_accum.handle,
        linear=v_linear.handle,
        lr=lr_val,
        l1=l1_val,
        l2=l2_val,
        l2_shrinkage=l2s_val,
        lr_power=lrp_val,
        grad=tf.constant(grad),
        indices=to_idx(indices),
        use_locking=use_locking,
        multiply_linear_by_lr=False,
    )

    return (
        v_var.read_value().numpy(),
        v_accum.read_value().numpy(),
        v_linear.read_value().numpy(),
    )


def sparse_apply_ftrl_v2_golden(
    var,
    accum,
    linear,
    grad,
    indices,
    lr,
    l1,
    l2,
    l2_shrinkage,
    lr_power,
    *,
    use_locking=False,
    **kwargs,
):
    """
    Kernel golden for sparse_apply_ftrl_v2 via tf.raw_ops.ResourceSparseApplyFtrlV2.

    For float16/bfloat16 inputs, computation is performed in float32 and then cast back.

    Parameters follow @sparse_apply_ftrl_v2_def.cpp definition order.

    Returns a tuple of numpy.ndarray: (var_out, accum_out, linear_out)
    """
    orig_dtype = var.dtype

    if indices.size == 0:
        return var, accum, linear

    # README: rows whose index is out of [0, N) are skipped. TF rejects such
    # indices, so drop them here using plain Python selection (no NumPy masking)
    # before calling the TF reference.
    num_rows = var.shape[0]
    keep = [k for k in range(indices.shape[0]) if 0 <= int(indices[k]) < num_rows]
    if not keep:
        return var, accum, linear
    if len(keep) != indices.shape[0]:
        indices = np.asarray([int(indices[k]) for k in keep], dtype=indices.dtype)
        grad = np.stack([grad[k] for k in keep], axis=0)

    is_fp16 = str(orig_dtype) == "float16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "float16"
    )
    is_bf16 = str(orig_dtype) == "bfloat16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "bfloat16"
    )
    needs_upcast = is_fp16 or is_bf16
    compute_dtype = np.float32 if needs_upcast else orig_dtype

    def safe_cast(arr, target_dtype):
        if arr.dtype != target_dtype:
            return arr.astype(target_dtype)
        return arr.copy()

    var_c = safe_cast(var, compute_dtype)
    accum_c = safe_cast(accum, compute_dtype)
    linear_c = safe_cast(linear, compute_dtype)
    grad_c = safe_cast(grad, compute_dtype)
    lr_c = safe_cast(np.asarray(lr), compute_dtype)
    l1_c = safe_cast(np.asarray(l1), compute_dtype)
    l2_c = safe_cast(np.asarray(l2), compute_dtype)
    l2_shrinkage_c = safe_cast(np.asarray(l2_shrinkage), compute_dtype)
    lr_power_c = safe_cast(np.asarray(lr_power), compute_dtype)

    out_var, out_accum, out_linear = _run_tf_ftrl_v2(
        var_c,
        accum_c,
        linear_c,
        grad_c,
        indices,
        lr_c,
        l1_c,
        l2_c,
        l2_shrinkage_c,
        lr_power_c,
        use_locking,
    )

    if needs_upcast:
        out_var = out_var.astype(orig_dtype, copy=False)
        out_accum = out_accum.astype(orig_dtype, copy=False)
        out_linear = out_linear.astype(orig_dtype, copy=False)

    return out_var, out_accum, out_linear


class _TfSparseApplyFtrlV2:
    """TensorFlow SparseApplyFtrlV2 reference computed in native dtype (third-party baseline)."""

    def __init__(self, *, use_locking=False, **kwargs):
        self.use_locking = bool(use_locking)

    def __call__(
        self,
        var,
        accum,
        linear,
        grad,
        indices,
        lr,
        l1,
        l2,
        l2_shrinkage,
        lr_power,
        **kwargs,
    ):
        def to_np(value):
            if hasattr(value, "numpy"):
                try:
                    return value.numpy()
                except (TypeError, RuntimeError):
                    pass
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            return np.asarray(value)

        return _run_tf_ftrl_v2(
            to_np(var),
            to_np(accum),
            to_np(linear),
            to_np(grad),
            to_np(indices),
            to_np(lr),
            to_np(l1),
            to_np(l2),
            to_np(l2_shrinkage),
            to_np(lr_power),
            self.use_locking,
        )


class _TorchSparseApplyFtrlV2:
    """PyTorch third-party baseline (GPU/CPU), mirroring the TF FtrlV2 formula.

    Runs on the remote XPU server, where only the torch provider is available.
    fp16/bf16 inputs are promoted to fp32 for the computation and cast back once
    at the end, matching the delivered TensorFlow golden.
    """

    def __init__(self, *, use_locking=False, **kwargs):
        self.use_locking = bool(use_locking)

    def __call__(
        self,
        var,
        accum,
        linear,
        grad,
        indices,
        lr,
        l1,
        l2,
        l2_shrinkage,
        lr_power,
        **kwargs,
    ):
        import torch

        def as_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value)

        var_t = as_tensor(var)
        out_dtype = var_t.dtype
        promote = out_dtype in (torch.float16, torch.bfloat16)

        def to_compute(t):
            return t.to(torch.float32) if promote else t

        v = to_compute(var_t).clone()
        a = to_compute(as_tensor(accum)).clone()
        lin = to_compute(as_tensor(linear)).clone()
        g = to_compute(as_tensor(grad))
        idx = as_tensor(indices).reshape(-1)

        def scalar(value):
            return float(as_tensor(value).reshape(-1)[0].item())

        lr_v = scalar(lr)
        l1_v = scalar(l1)
        l2_v = scalar(l2)
        l2s_v = scalar(l2_shrinkage)
        lrp_v = scalar(lr_power)
        n_rows = v.shape[0]

        with torch.no_grad():
            for pos in range(idx.numel()):
                row = int(idx[pos].item())
                if row < 0 or row >= n_rows:
                    continue
                v_row = v[row]
                a_row = a[row]
                g_row = g[pos]
                grad_shrink = g_row + (2.0 * l2s_v) * v_row if l2s_v != 0.0 else g_row
                old_pow = torch.pow(a_row, -lrp_v)
                new_acc = a_row + g_row * g_row
                new_pow = torch.pow(new_acc, -lrp_v)
                a[row] = new_acc
                delta = (old_pow - new_pow) / lr_v
                new_lin = lin[row] + grad_shrink + delta * v_row
                lin[row] = new_lin
                l1_adj = torch.clamp(new_lin, min=-l1_v, max=l1_v)
                v[row] = (l1_adj - new_lin) / (new_pow / lr_v + 2.0 * l2_v)

        if promote:
            v = v.to(out_dtype)
            a = a.to(out_dtype)
            lin = lin.to(out_dtype)
        return [v, a, lin]


class SparseApplyFtrlV2TestSpec:
    """SparseApplyFtrlV2 kernel TestSpec (GEIR pathway)."""

    golden = staticmethod(sparse_apply_ftrl_v2_golden)
    third_party: ClassVar[dict] = {
        "torch": _TorchSparseApplyFtrlV2,
        "tf": _TfSparseApplyFtrlV2,
    }
