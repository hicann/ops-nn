#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

__golden__ = {
    "kernel": {"sparse_apply_proximal_adagrad": "sparse_apply_proximal_adagrad_golden"}
}

__input__ = {
    "kernel": {"sparse_apply_proximal_adagrad": "sparse_apply_proximal_adagrad_input"}
}


def sparse_apply_proximal_adagrad_input(
    var, accum, lr, l1, l2, grad, indices, **kwargs
):
    """
    Custom input generator that ensures indices are unique and accum is positive.
    Per SE doc C8: indices values must be unique, otherwise results are unpredictable.
    Per SE doc P4: accum must be > 0 (no epsilon in denominator).

    Strategy:
    - indices: unique, within [0, N) (best-effort; only valid in-range indices are deduped)
    - accum: clip to [0.1, Inf)
    - lr: ensure > 0
    - l1: clip to >= 0
    - l2: clip to >= 0
    """
    N = var.shape[0]
    M = indices.shape[0]

    accum = np.where(np.isfinite(accum) & (accum > 0), accum, accum.dtype.type(2.0))
    var = np.where(np.isfinite(var), var, var.dtype.type(0.0))
    grad = np.where(np.isfinite(grad), grad, grad.dtype.type(0.0))
    lr = np.maximum(lr, lr.dtype.type(1e-6))
    l1 = np.maximum(l1, l1.dtype.type(0.0))
    l2 = np.maximum(l2, l2.dtype.type(0.0))

    if M > 0 and N > 0:
        valid_mask = (indices >= 0) & (indices < N)
        valid_count = int(np.sum(valid_mask))

        if valid_count > 0 and valid_count <= N:
            unique_valid = np.random.choice(N, size=valid_count, replace=False).astype(
                indices.dtype
            )
            indices = indices.copy()
            indices[valid_mask] = unique_valid

    return [var, accum, lr, l1, l2, grad, indices]


def sparse_apply_proximal_adagrad_golden(
    var, accum, lr, l1, l2, grad, indices, *, use_locking=False, **kwargs
):
    """
    Golden function for sparse_apply_proximal_adagrad.

    All the parameters (names and order) follow SE doc prototype definition without outputs.
    All the input Tensors are numpy.ndarray.

    Implementation uses tf.raw_ops.ResourceSparseApplyProximalAdagrad for accuracy.
    For float16/bfloat16 inputs, computation is performed in float32 and then cast back.

    Args:
        var:     np.ndarray, shape [N, D1, D2, ...], variable to be updated
        accum:   np.ndarray, shape [N, D1, D2, ...], gradient-squared accumulator
        lr:      np.ndarray or float, scalar learning rate
        l1:      np.ndarray or float, scalar L1 regularization coefficient
        l2:      np.ndarray or float, scalar L2 regularization coefficient
        grad:    np.ndarray, shape [M, D1, D2, ...], sparse gradient
        indices: np.ndarray, shape [M], sparse indices (int32 or int64)
        use_locking: bool, whether to update using locks (default False)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        (var_out, accum_out): tuple of numpy.ndarray, updated var and accum

    Note:
        SparseApplyProximalAdagrad denominator does NOT include epsilon.
        accum must be initialized to a positive value (e.g., 1.0) to avoid division by zero.
        OOB indices are silently filtered out (consistent with Ascend canndev behavior).
    """
    import tensorflow as tf

    orig_dtype = var.dtype

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
        return arr

    var_c = safe_cast(var, compute_dtype)
    accum_c = safe_cast(accum, compute_dtype)
    lr_c = safe_cast(lr, compute_dtype)
    l1_c = safe_cast(l1, compute_dtype)
    l2_c = safe_cast(l2, compute_dtype)
    grad_c = safe_cast(grad, compute_dtype)

    # Filter OOB indices (consistent with Ascend SIMT operator behavior)
    N = var.shape[0]
    valid_mask = (indices >= 0) & (indices < N)
    if not np.all(valid_mask):
        if np.any(valid_mask):
            indices = indices[valid_mask]
            grad_c = grad_c[valid_mask]
        else:
            # All indices OOB: no updates, return original var/accum
            out_var = var.copy()
            out_accum = accum.copy()
            if needs_upcast:
                out_var = out_var.astype(orig_dtype, copy=False)
                out_accum = out_accum.astype(orig_dtype, copy=False)
            return out_var, out_accum

    if indices.size == 0:
        out_var = var.copy()
        out_accum = accum.copy()
        if needs_upcast:
            out_var = out_var.astype(orig_dtype, copy=False)
            out_accum = out_accum.astype(orig_dtype, copy=False)
        return out_var, out_accum

    v_var = tf.Variable(var_c, name="var")
    v_accum = tf.Variable(accum_c, name="accum")

    lr_val = lr_c.item() if hasattr(lr_c, "item") else float(lr_c)
    l1_val = l1_c.item() if hasattr(l1_c, "item") else float(l1_c)
    l2_val = l2_c.item() if hasattr(l2_c, "item") else float(l2_c)

    tf.raw_ops.ResourceSparseApplyProximalAdagrad(
        var=v_var.handle,
        accum=v_accum.handle,
        lr=lr_val,
        l1=l1_val,
        l2=l2_val,
        grad=tf.constant(grad_c),
        indices=tf.constant(indices),
        use_locking=use_locking,
    )

    out_var = v_var.read_value().numpy()
    out_accum = v_accum.read_value().numpy()

    if needs_upcast:
        out_var = out_var.astype(orig_dtype, copy=False)
        out_accum = out_accum.astype(orig_dtype, copy=False)

    return out_var, out_accum
