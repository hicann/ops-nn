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
import tensorflow as tf

__golden__ = {"kernel": {"sparse_apply_rms_prop": "sparse_apply_rms_prop_golden"}}

__input__ = {"kernel": {"sparse_apply_rms_prop": "sparse_apply_rms_prop_input"}}


def _constrain(arr, fn):
    """Apply constraint in fp32, then cast back to original dtype."""
    orig_dtype = arr.dtype
    result = fn(arr.astype(np.float32))
    return result.astype(orig_dtype)


def sparse_apply_rms_prop_input(
    var, ms, mom, lr, rho, momentum, epsilon, grad, indices, **kwargs
):
    """
    Custom input generator: enforce valid parameter ranges for RMSProp semantics.

    Constraints:
      - indices: unique, within [0, N) (best-effort; only valid in-range indices are deduped)
      - ms >= 0 (mean square accumulator, prevents NaN in sqrt)
      - lr > 0, rho in (0, 1], epsilon > 0, momentum >= 0
    """
    N = var.shape[0]
    M = indices.shape[0]

    if M > 0 and N > 0:
        valid_mask = (indices >= 0) & (indices < N)
        valid_count = int(np.sum(valid_mask))
        if valid_count > 0 and valid_count <= N:
            unique_valid = np.random.choice(N, size=valid_count, replace=False).astype(
                indices.dtype
            )
            indices = indices.copy()
            indices[valid_mask] = unique_valid

    lr = _constrain(lr, lambda x: np.maximum(np.abs(x), 1e-6))
    rho = _constrain(rho, lambda x: np.clip(np.abs(x), 0.01, 1.0))
    momentum = _constrain(momentum, np.abs)
    epsilon = _constrain(epsilon, lambda x: np.maximum(np.abs(x), 1e-8))
    ms = _constrain(ms, np.abs)

    return [var, ms, mom, lr, rho, momentum, epsilon, grad, indices]


def sparse_apply_rms_prop_golden(
    var,
    ms,
    mom,
    lr,
    rho,
    momentum,
    epsilon,
    grad,
    indices,
    *,
    use_locking=False,
    **kwargs,
):
    """
    Kernel golden for sparse_apply_rms_prop via tf.raw_ops.ResourceSparseApplyRMSProp.

    For float16/bfloat16 inputs, computation is performed in float32 and then cast back.

    Parameters follow @sparse_apply_rms_prop_def.cpp definition order.

    Returns a tuple of numpy.ndarray: (var_out, ms_out, mom_out)
    """
    orig_dtype = var.dtype

    if indices.size == 0:
        return var, ms, mom

    def to_idx(arr):
        return tf.constant(arr, dtype=tf.int32 if arr.dtype == np.int32 else tf.int64)

    is_fp16 = str(orig_dtype) == "float16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "float16"
    )
    is_bf16 = str(orig_dtype) == "bfloat16" or (
        hasattr(orig_dtype, "name") and orig_dtype.name == "bfloat16"
    )
    needs_upcast = is_fp16 or is_bf16
    compute_dtype = np.float32 if needs_upcast else orig_dtype

    # Filter OOB indices (TF's ResourceSparseApplyRMSProp rejects them)
    N_var = var.shape[0]
    valid_mask = (indices >= 0) & (indices < N_var)
    if not np.all(valid_mask):
        if np.any(valid_mask):
            indices = indices[valid_mask]
            grad = grad[valid_mask]
        else:
            return var, ms, mom

    if indices.size == 0:
        return var, ms, mom

    def safe_cast(arr, target_dtype):
        if arr.dtype != target_dtype:
            return arr.astype(target_dtype)
        return arr.copy()

    var_c = safe_cast(var, compute_dtype)
    ms_c = safe_cast(ms, compute_dtype)
    mom_c = safe_cast(mom, compute_dtype)
    grad_c = safe_cast(grad, compute_dtype)
    lr_c = safe_cast(lr, compute_dtype)
    rho_c = safe_cast(rho, compute_dtype)
    momentum_c = safe_cast(momentum, compute_dtype)
    epsilon_c = safe_cast(epsilon, compute_dtype)

    v_var = tf.Variable(var_c, name="var")
    v_ms = tf.Variable(ms_c, name="ms")
    v_mom = tf.Variable(mom_c, name="mom")

    lr_val = lr_c.item() if hasattr(lr_c, "item") else float(lr_c)
    rho_val = rho_c.item() if hasattr(rho_c, "item") else float(rho_c)
    momentum_val = (
        momentum_c.item() if hasattr(momentum_c, "item") else float(momentum_c)
    )
    epsilon_val = epsilon_c.item() if hasattr(epsilon_c, "item") else float(epsilon_c)

    tf.raw_ops.ResourceSparseApplyRMSProp(
        var=v_var.handle,
        ms=v_ms.handle,
        mom=v_mom.handle,
        lr=lr_val,
        rho=rho_val,
        momentum=momentum_val,
        epsilon=epsilon_val,
        grad=tf.constant(grad_c),
        indices=to_idx(indices),
        use_locking=use_locking,
    )

    out_var = v_var.read_value().numpy()
    out_ms = v_ms.read_value().numpy()
    out_mom = v_mom.read_value().numpy()

    if needs_upcast:
        out_var = out_var.astype(orig_dtype, copy=False)
        out_ms = out_ms.astype(orig_dtype, copy=False)
        out_mom = out_mom.astype(orig_dtype, copy=False)

    return out_var, out_ms, out_mom
