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

import hashlib
from typing import ClassVar

import numpy as np
import torch

__spec__ = {
    "inplace_sub": "InplaceSubKernelSpec",
    "tf.raw_ops.InplaceSub": "InplaceSubTfSpec",
}
__golden__ = {
    "kernel": {"inplace_sub": "inplace_sub_golden"},
    "e2e": {"tf.raw_ops.InplaceSub": "inplace_sub_tf_golden"},
}
__input__ = {
    "kernel": {"inplace_sub": "inplace_sub_input"},
    "e2e": {"tf.raw_ops.InplaceSub": "inplace_sub_tf_input"},
}
_E2E_ORIGINAL_X = {}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "bfloat16": {"standard": "cross_check", "level": "L1"},
    "int8": {"standard": "binary_equal"},
    "int16": {"standard": "binary_equal"},
    "int32": {"standard": "binary_equal"},
    "int64": {"standard": "binary_equal"},
    "uint8": {"standard": "binary_equal"},
    "uint16": {"standard": "binary_equal"},
    "uint32": {"standard": "binary_equal"},
    "uint64": {"standard": "binary_equal"},
}


def _rng_for_case(testcase_name):
    digest = hashlib.md5(testcase_name.encode("utf-8"), usedforsecurity=False).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _supports_special_values(array):
    dtype_name = str(array.dtype)
    return (
        "bfloat16" in dtype_name
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    )


def inplace_sub_input(x, indices, v, **kwargs):
    testcase_name = kwargs.get("testcase_name", "")
    if "dfx_nan" in testcase_name and _supports_special_values(x):
        x = np.array(x, copy=True)
        v = np.array(v, copy=True)
        if x.size > 0:
            x.reshape(-1)[0] = np.nan
        if v.size > 0:
            v.reshape(-1)[0] = np.nan
    elif "dfx_inf" in testcase_name and _supports_special_values(x):
        x = np.array(x, copy=True)
        v = np.array(v, copy=True)
        if x.size > 0:
            x.reshape(-1)[0] = np.inf
        if v.size > 0:
            v.reshape(-1)[0] = -np.inf

    if indices.size == 0 or x.shape[0] == 0:
        return [x, indices, v]
    indices_shape = indices.shape
    if "dfx_negative_index" in testcase_name:
        values = np.array([-1, -2, -x.shape[0]], dtype=indices.dtype)
        indices = np.resize(values, indices.size).astype(indices.dtype)
        return [x, indices.reshape(indices_shape), v]
    if "dfx_wrapped_index" in testcase_name:
        values = np.array(
            [x.shape[0], x.shape[0] + 1, 2 * x.shape[0] - 1], dtype=indices.dtype
        )
        indices = np.resize(values, indices.size).astype(indices.dtype)
        return [x, indices.reshape(indices_shape), v]
    if indices.size > x.shape[0]:
        raise ValueError(
            "InplaceSub positive cases require indices length not greater than x.shape[0]."
        )
    rng = _rng_for_case(testcase_name)
    indices = rng.choice(x.shape[0], indices.size, replace=False).astype(indices.dtype)
    return [x, indices.reshape(indices_shape), v]


def _prepare_updates(updates, idx, y_rank):
    if updates.ndim == idx.ndim and y_rank > 1:
        return updates.reshape(updates.shape + (1,) * (y_rank - 1))
    return updates


def _is_wide_unsigned(dtype):
    return dtype in (np.dtype("uint16"), np.dtype("uint32"), np.dtype("uint64"))


def _numpy_to_torch_tensor(array):
    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _prepare_inplace_sub_inputs(x, indices, v):
    if isinstance(x, torch.Tensor):
        return_torch = True
        result_dtype = x.dtype
        y_source = x.clone()
        updates_source = v.clone()
        if result_dtype in (torch.uint16, torch.uint32, torch.uint64):
            y = y_source.to(torch.int64)
            updates = updates_source.to(torch.int64)
        else:
            y = y_source
            updates = updates_source
        idx_array = indices.to(torch.int64)
    else:
        return_torch = False
        x_array = np.array(x, copy=True)
        updates_array = np.array(v, copy=True)
        result_dtype = x_array.dtype
        if _is_wide_unsigned(result_dtype):
            y = torch.from_numpy(x_array.astype(np.int64))
            updates = torch.from_numpy(updates_array.astype(np.int64))
        else:
            y = _numpy_to_torch_tensor(x_array)
            updates = _numpy_to_torch_tensor(updates_array)
        idx_array = torch.from_numpy(indices.astype(np.int64))
    if y.shape[0] > 0:
        idx_array = ((idx_array % y.shape[0]) + y.shape[0]) % y.shape[0]
    return y, idx_array, updates, result_dtype, return_torch


def _restore_inplace_sub_output(y, result_dtype, return_torch):
    if return_torch:
        return y.to(result_dtype)
    if _is_wide_unsigned(result_dtype):
        return y.numpy().astype(result_dtype)
    return _torch_to_numpy_tensor(y)


def _inplace_sub_golden_compute(x, indices, v):
    y, idx, updates, result_dtype, return_torch = _prepare_inplace_sub_inputs(
        x, indices, v
    )
    updates = _prepare_updates(updates, idx, y.dim())
    update_shape = (idx.numel(),) + tuple(y.shape[1:])
    updates = updates.expand(update_shape)
    y = torch.index_add(y, 0, idx.reshape(-1), -updates.reshape(update_shape))
    return _restore_inplace_sub_output(y, result_dtype, return_torch)


class _InplaceSubTfCompose:
    def __call__(self, x, indices, v, **kwargs):
        import tensorflow as tf

        y = tf.convert_to_tensor(x)
        idx = tf.reshape(tf.cast(tf.convert_to_tensor(indices), tf.int32), [-1])
        updates = tf.cast(tf.convert_to_tensor(v), y.dtype)

        def no_op():
            return tf.identity(y)

        def subtract():
            wrapped = tf.math.floormod(idx, tf.shape(y, out_type=tf.int32)[0])
            return tf.tensor_scatter_nd_sub(y, tf.reshape(wrapped, [-1, 1]), updates)

        return [tf.cond(tf.equal(tf.size(idx), 0), no_op, subtract)]


class _InplaceSubTorchCompose:
    """Independent Torch provider using index_put_ accumulation semantics."""

    def __call__(self, x, indices, v, **kwargs):
        del kwargs
        y, idx, updates, result_dtype, return_torch = _prepare_inplace_sub_inputs(
            x, indices, v
        )
        if idx.numel() == 0 or y.shape[0] == 0:
            return [_restore_inplace_sub_output(y, result_dtype, return_torch)]
        updates = _prepare_updates(updates, idx, y.dim())
        update_shape = (idx.numel(),) + tuple(y.shape[1:])
        updates = updates.expand(update_shape)
        # index_put_ with accumulate=True is independent of the kernel golden
        # (which uses index_add) while preserving duplicate-index behavior.
        y.index_put_((idx.reshape(-1),), -updates, accumulate=True)
        return [_restore_inplace_sub_output(y, result_dtype, return_torch)]


class _InplaceSubTfTorchCompose:
    def __call__(self, x, i, v, **kwargs):
        return _InplaceSubTorchCompose.__call__(self, x, i, v, **kwargs)


def _promote_tf_golden_dtype(tensor):
    import tensorflow as tf

    promote = {
        tf.float16: tf.float32,
        tf.bfloat16: tf.float32,
        tf.float32: tf.float64,
        tf.complex64: tf.complex128,
    }
    return tf.cast(tensor, promote.get(tensor.dtype, tensor.dtype))


def _inplace_sub_tf_api_golden(x, i, v):
    import tensorflow as tf

    input_tensor = tf.convert_to_tensor(x)
    input_dtype = input_tensor.dtype
    work = _promote_tf_golden_dtype(input_tensor)
    indices = tf.reshape(tf.cast(tf.convert_to_tensor(i), tf.int32), [-1])
    updates = tf.cast(tf.convert_to_tensor(v), work.dtype)

    def no_op():
        return tf.identity(work)

    def subtract():
        wrapped = tf.math.floormod(indices, tf.shape(work, out_type=tf.int32)[0])
        return tf.tensor_scatter_nd_sub(work, tf.reshape(wrapped, [-1, 1]), updates)

    is_no_op = tf.logical_or(tf.equal(tf.size(indices), 0), tf.equal(tf.size(work), 0))
    result = tf.cond(is_no_op, no_op, subtract)
    return tf.cast(result, input_dtype)


def inplace_sub_tf_input(x, i, v, testcase_name=None, **kwargs):
    """Preserve x before TF raw-op execution for the custom E2E golden.

    TTK currently tracks explicit inplace indexes only in positional args,
    while TensorFlow raw ops are dispatched through keyword args. Keeping the
    pre-execution value here prevents golden generation from observing the
    already-mutated Variable.
    """
    del i, v, kwargs
    if testcase_name:
        value = x.numpy() if hasattr(x, "numpy") else x
        _E2E_ORIGINAL_X[testcase_name] = np.array(value, copy=True)


class InplaceSubKernelSpec:
    @staticmethod
    def golden(x, indices, v, **kwargs):
        return [_inplace_sub_golden_compute(x, indices, v)]

    third_party: ClassVar[dict] = {
        "torch": _InplaceSubTorchCompose,
        "tf": _InplaceSubTfCompose,
    }
    customize_inputs = staticmethod(inplace_sub_input)
    tolerance = _TOL


class InplaceSubTfSpec:
    @staticmethod
    def golden(x, i, v, testcase_name=None, **kwargs):
        original_x = _E2E_ORIGINAL_X.pop(testcase_name, None)
        return [
            _inplace_sub_tf_api_golden(
                original_x if original_x is not None else x, i, v
            )
        ]

    customize_inputs = staticmethod(inplace_sub_tf_input)
    third_party: ClassVar[dict] = {"torch": _InplaceSubTfTorchCompose}
    tolerance = _TOL


def inplace_sub_golden(x, indices, v, **kwargs):
    return InplaceSubKernelSpec.golden(x, indices, v, **kwargs)


def inplace_sub_tf_golden(x, i, v, **kwargs):
    return InplaceSubTfSpec.golden(x, i, v, **kwargs)
