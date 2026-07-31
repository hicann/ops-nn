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

"""
Competitor Golden script for sparse_apply_ftrl (TensorFlow implementation).

This file provides a TensorFlow-native golden that calls
``tf.raw_ops.ResourceSparseApplyFtrl`` as the reference implementation,
complementing the numpy-based ``golden.py``. Both files share the same
function signature and ``__golden__`` / ``__input__`` declarations so that
they are interchangeable from the TTK framework perspective.

Note:
    ``SparseApplyFtrl`` (ref-based) does not support eager execution in TF 2.x.
    ``ResourceSparseApplyFtrl`` (resource-based) works in eager mode when the
    Variable handles are passed explicitly via ``var.handle``.
"""

import numpy as np
import tensorflow as tf

__golden__ = {"kernel": {"sparse_apply_ftrl": "sparse_apply_ftrl_golden"}}

__input__ = {"kernel": {"sparse_apply_ftrl": "sparse_apply_ftrl_input"}}


def sparse_apply_ftrl_input(
    var, accum, linear, grad, indices, lr, l1, l2, lr_power, **kwargs
):
    """
    Custom input plugin for sparse_apply_ftrl.
    Ensures indices array contains NO duplicate values by using
    deterministic sequential indices [0, 1, 2, ..., N-1].
    """
    if indices is not None and indices.size > 0:
        N = indices.size
        new_indices = np.arange(N, dtype=indices.dtype)
        indices_flat = indices.flatten()
        indices_flat[:] = new_indices
        indices[:] = indices_flat.reshape(indices.shape)
    return [var, accum, linear, grad, indices, lr, l1, l2, lr_power]


def _to_scalar(value):
    """Extract a Python float from a numpy scalar / 0-D / 1-D array / Python float."""
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def sparse_apply_ftrl_golden(
    var,
    accum,
    linear,
    grad,
    indices,
    lr,
    l1,
    l2,
    lr_power,
    *,
    use_locking=False,
    **kwargs,
):
    """
    Competitor Golden function for sparse_apply_ftrl (TensorFlow implementation).

    Uses ``tf.raw_ops.ResourceSparseApplyFtrl`` as the reference, which is the
    TensorFlow native FTRL-Proximal sparse update op. This serves as an
    independent cross-check against the numpy-based golden in ``golden.py``.

    All the parameters (names and order) follow @sparse_apply_ftrl_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        var: np.ndarray, shape (D0, D1, ...), float32 - variable to update
        accum: np.ndarray, shape (D0, D1, ...), float32 - gradient squared accumulator
        linear: np.ndarray, shape (D0, D1, ...), float32 - linear accumulator
        grad: np.ndarray, shape (N, D1, ...), float32 - gradient tensor
        indices: np.ndarray, shape (N,), int32/int64 - index vector
        lr: np.ndarray (0-D or 1-D) or float, float32 - learning rate scalar
        l1: np.ndarray (0-D or 1-D) or float, float32 - L1 regularization coefficient
        l2: np.ndarray (0-D or 1-D) or float, float32 - L2 regularization coefficient
        lr_power: np.ndarray (0-D or 1-D) or float, float32 - learning rate power
        use_locking: bool - whether to use locking (passed to TF op)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Tuple of (var_out, accum_out, linear_out) updated tensors (np.ndarray, float32)
    """
    # Ensure float32 for variable tensors
    var_np = np.asarray(var, dtype=np.float32)
    accum_np = np.asarray(accum, dtype=np.float32)
    linear_np = np.asarray(linear, dtype=np.float32)
    grad_np = np.asarray(grad, dtype=np.float32)
    indices_np = np.asarray(indices)

    # Extract scalar values (handle 0-D, 1-D ndarray and Python float)
    lr_val = _to_scalar(lr)
    l1_val = _to_scalar(l1)
    l2_val = _to_scalar(l2)
    lr_power_val = _to_scalar(lr_power)

    # Wrap in tf.Variable for in-place (resource) semantics
    var_v = tf.Variable(var_np)
    accum_v = tf.Variable(accum_np)
    linear_v = tf.Variable(linear_np)

    grad_t = tf.constant(grad_np, dtype=tf.float32)
    indices_t = tf.constant(indices_np)
    lr_t = tf.constant(lr_val, dtype=tf.float32)
    l1_t = tf.constant(l1_val, dtype=tf.float32)
    l2_t = tf.constant(l2_val, dtype=tf.float32)
    lr_power_t = tf.constant(lr_power_val, dtype=tf.float32)

    # Call TF native FTRL op (resource-based, works in eager mode via .handle)
    tf.raw_ops.ResourceSparseApplyFtrl(
        var=var_v.handle,
        accum=accum_v.handle,
        linear=linear_v.handle,
        grad=grad_t,
        indices=indices_t,
        lr=lr_t,
        l1=l1_t,
        l2=l2_t,
        lr_power=lr_power_t,
        use_locking=bool(use_locking),
    )

    # Read back updated values
    var_out = var_v.numpy()
    accum_out = accum_v.numpy()
    linear_out = linear_v.numpy()

    return var_out, accum_out, linear_out
