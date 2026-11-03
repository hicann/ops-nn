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

__spec__ = {
    "inplace_sub": "InplaceSubKernelSpec",
    "tf.raw_ops.InplaceSub": "InplaceSubTfSpec",
}
__golden__ = {
    "kernel": {"inplace_sub": "inplace_sub_golden"},
}
__input__ = {
    "kernel": {"inplace_sub": "inplace_sub_input"},
}

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
    "complex32": {"standard": "binary_equal"},
    "complex64": {"standard": "binary_equal"},
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
    # Invalid-shape cases must reach the operator unchanged.
    if (
        x.ndim == 0
        or indices.ndim != 1
        or v.ndim != x.ndim
        or v.shape[0] != indices.size
        or v.shape[1:] != x.shape[1:]
        or indices.size > x.shape[0]
    ):
        return [x, indices, v]
    if "dfx_signed_overflow" in testcase_name:
        x = np.array(x, copy=True)
        indices = np.zeros_like(indices)
        v = np.array(v, copy=True)
        dtype_info = np.iinfo(x.dtype)
        x.reshape(-1)[:] = np.resize(
            np.array(
                [dtype_info.max, dtype_info.min, dtype_info.max, dtype_info.min],
                dtype=x.dtype,
            ),
            x.size,
        )
        v.reshape(-1)[:] = np.resize(np.array([-1, 1, -2, 2], dtype=v.dtype), v.size)
        return [x, indices, v]
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
    import torch

    if "bfloat16" in array.dtype.name:
        return torch.from_numpy(array.view(dtype=np.int16)).view(torch.bfloat16)
    return torch.from_numpy(array)


def _torch_to_numpy_tensor(tensor):
    import torch

    if tensor.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16

        return tensor.view(torch.int16).numpy().view(dtype=bfloat16)
    return tensor.numpy()


def _prepare_inplace_sub_inputs(x, indices, v):
    import torch

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
    """Use TensorFlow for both CPU paths without overriding TTK Promote."""
    import tensorflow as tf

    y = tf.convert_to_tensor(x)
    updates = tf.convert_to_tensor(v)
    idx = tf.reshape(tf.convert_to_tensor(indices), [-1])
    if int(tf.size(y)) == 0 or int(tf.size(idx)) == 0:
        return y.numpy()
    idx = tf.math.floormod(idx, tf.shape(y, out_type=idx.dtype)[0])
    updates = tf.reshape(updates, (int(tf.size(idx)),) + tuple(y.shape[1:]))
    return tf.tensor_scatter_nd_sub(y, tf.reshape(idx, [-1, 1]), updates).numpy()


class _InplaceSubTorchCompose:
    """Independent Torch provider using index_put_ accumulation semantics."""

    def __call__(self, x, indices, v, **kwargs):
        import torch

        del kwargs
        y, idx, updates, result_dtype, return_torch = _prepare_inplace_sub_inputs(
            x, indices, v
        )
        if idx.numel() == 0 or y.shape[0] == 0:
            return [_restore_inplace_sub_output(y, result_dtype, return_torch)]
        updates = _prepare_updates(updates, idx, y.dim())
        update_shape = (idx.numel(),) + tuple(y.shape[1:])
        updates = updates.expand(update_shape)
        # Accumulation preserves duplicate-index behavior independently of the
        # TensorFlow scatter used by the CPU golden.
        if y.is_complex():
            # Keep the components independent: CUDA complex accumulation can
            # introduce cross-component NaNs for infinite updates.
            torch.view_as_real(y).index_put_(
                (idx.reshape(-1),), -torch.view_as_real(updates), accumulate=True
            )
        else:
            y.index_put_((idx.reshape(-1),), -updates, accumulate=True)
        return [_restore_inplace_sub_output(y, result_dtype, return_torch)]


class _InplaceSubTfTorchCompose:
    def __call__(self, x, i, v, **kwargs):
        return _InplaceSubTorchCompose.__call__(self, x, i, v, **kwargs)


def _inplace_sub_tf_api_golden(x, i, v):
    return _inplace_sub_golden_compute(x, i, v)


class InplaceSubKernelSpec:
    @staticmethod
    def golden(x, indices, v, **kwargs):
        return [_inplace_sub_golden_compute(x, indices, v)]

    third_party: ClassVar[dict] = {"torch": _InplaceSubTorchCompose}
    customize_inputs = staticmethod(inplace_sub_input)
    tolerance = _TOL


class InplaceSubTfSpec:
    # The native TF CPU fallback applies Promote to fresh raw inputs.
    # TF InplaceSub also changes ordinary Tensor storage: repeated E2E calls
    # require a fresh x; use one call without warmup for functional validation.
    golden = None
    third_party: ClassVar[dict] = {"torch": _InplaceSubTfTorchCompose}
    tolerance = _TOL


def inplace_sub_golden(x, indices, v, **kwargs):
    return InplaceSubKernelSpec.golden(x, indices, v, **kwargs)


def inplace_sub_tf_golden(x, i, v, **kwargs):
    return [_inplace_sub_tf_api_golden(x, i, v)]
