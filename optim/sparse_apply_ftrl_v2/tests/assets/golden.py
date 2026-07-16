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

__golden__ = {"kernel": {"sparse_apply_ftrl_v2": "sparse_apply_ftrl_v2_golden"}}

__input__ = {"kernel": {"sparse_apply_ftrl_v2": "sparse_apply_ftrl_v2_input"}}


def _constrain(arr, fn):
    """Apply constraint in fp32, then cast back to original dtype."""
    orig_dtype = arr.dtype
    result = fn(arr.astype(np.float32))
    return result.astype(orig_dtype)


def sparse_apply_ftrl_v2_input(
    var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power, **kwargs
):
    """
    Custom input generator: enforce valid parameter ranges for Ftrl semantics.

    Constraints:
      - indices: unique, within [0, N) (best-effort; only valid in-range indices are deduped)
      - lr > 0, l1 >= 0, l2 >= 0, l2_shrinkage >= 0
      - lr_power <= 0
      - accum >= 0 (squared gradient accumulator, prevents NaN in pow/sqrt)
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
    l1 = _constrain(l1, np.abs)
    l2 = _constrain(l2, np.abs)
    l2_shrinkage = _constrain(l2_shrinkage, np.abs)
    lr_power = _constrain(lr_power, lambda x: -np.abs(x))
    accum = _constrain(accum, np.abs)

    return [var, accum, linear, grad, indices, lr, l1, l2, l2_shrinkage, lr_power]


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

    # Filter OOB indices (TF's ResourceSparseApplyFtrlV2 rejects them)
    N_var = var.shape[0]
    valid_mask = (indices >= 0) & (indices < N_var)
    if not np.all(valid_mask):
        if np.any(valid_mask):
            indices = indices[valid_mask]
            grad = grad[valid_mask]
        else:
            return var, accum, linear

    if indices.size == 0:
        return var, accum, linear

    def safe_cast(arr, target_dtype):
        if arr.dtype != target_dtype:
            return arr.astype(target_dtype)
        return arr.copy()

    var_c = safe_cast(var, compute_dtype)
    accum_c = safe_cast(accum, compute_dtype)
    linear_c = safe_cast(linear, compute_dtype)
    grad_c = safe_cast(grad, compute_dtype)
    lr_c = safe_cast(lr, compute_dtype)
    l1_c = safe_cast(l1, compute_dtype)
    l2_c = safe_cast(l2, compute_dtype)
    l2_shrinkage_c = safe_cast(l2_shrinkage, compute_dtype)
    lr_power_c = safe_cast(lr_power, compute_dtype)

    v_var = tf.Variable(var_c, name="var")
    v_accum = tf.Variable(accum_c, name="accum")
    v_linear = tf.Variable(linear_c, name="linear")

    lr_val = lr_c.item() if hasattr(lr_c, "item") else float(lr_c)
    l1_val = l1_c.item() if hasattr(l1_c, "item") else float(l1_c)
    l2_val = l2_c.item() if hasattr(l2_c, "item") else float(l2_c)
    l2s_val = (
        l2_shrinkage_c.item()
        if hasattr(l2_shrinkage_c, "item")
        else float(l2_shrinkage_c)
    )
    lrp_val = lr_power_c.item() if hasattr(lr_power_c, "item") else float(lr_power_c)

    tf.raw_ops.ResourceSparseApplyFtrlV2(
        var=v_var.handle,
        accum=v_accum.handle,
        linear=v_linear.handle,
        lr=lr_val,
        l1=l1_val,
        l2=l2_val,
        l2_shrinkage=l2s_val,
        lr_power=lrp_val,
        grad=tf.constant(grad_c),
        indices=to_idx(indices),
        use_locking=use_locking,
        multiply_linear_by_lr=False,
    )

    out_var = v_var.read_value().numpy()
    out_accum = v_accum.read_value().numpy()
    out_linear = v_linear.read_value().numpy()

    if needs_upcast:
        out_var = out_var.astype(orig_dtype, copy=False)
        out_accum = out_accum.astype(orig_dtype, copy=False)
        out_linear = out_linear.astype(orig_dtype, copy=False)

    return out_var, out_accum, out_linear
