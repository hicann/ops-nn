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

import inspect

import numpy as np
import tensorflow as tf

__spec__ = {
    "scatter_non_aliasing_add": "ScatterNonAliasingAddKernelSpec",
    "tf.raw_ops.ScatterNdNonAliasingAdd": "TfScatterNdNonAliasingAddSpec",
}


def _scatter_nd_non_aliasing_add(input, indices, updates, bad_indices_policy=""):
    """Call both the TF 2.6 and newer raw-op signatures."""
    kwargs = {
        "input": input,
        "indices": indices,
        "updates": updates,
    }
    if (
        "bad_indices_policy"
        in inspect.signature(tf.raw_ops.ScatterNdNonAliasingAdd).parameters
    ):
        kwargs["bad_indices_policy"] = bad_indices_policy
    return tf.raw_ops.ScatterNdNonAliasingAdd(**kwargs)


def _numpy_scatter_add(x, indices, updates):
    y = np.array(x, copy=True)
    idx = np.asarray(indices)
    upd = np.asarray(updates)
    if y.size == 0 or idx.size == 0 or upd.size == 0:
        return y
    if idx.ndim == 1:
        idx = idx.reshape(1, -1)
    else:
        idx = idx.reshape(-1, idx.shape[-1])
    tail = y.shape[idx.shape[-1] :]
    upd = upd.reshape((idx.shape[0],) + tail)
    for i in range(idx.shape[0]):
        y[tuple(int(v) for v in idx[i])] += upd[i]
    return y


class ScatterNonAliasingAddKernelSpec:
    @staticmethod
    def golden(x, indices, updates, **kwargs):
        x = tf.convert_to_tensor(x)
        indices = tf.convert_to_tensor(indices)
        updates = tf.convert_to_tensor(updates)
        if (
            tf.size(x).numpy() == 0
            or tf.size(indices).numpy() == 0
            or tf.size(updates).numpy() == 0
        ):
            return x.numpy()
        if indices.shape[-1] > 5:
            return _numpy_scatter_add(x.numpy(), indices.numpy(), updates.numpy())
        if indices.shape.rank == 1:
            index_depth = indices.shape[-1]
            indices = tf.reshape(indices, (1, index_depth))
            updates = tf.reshape(updates, (1,) + tuple(x.shape[index_depth:]))
        out = tf.raw_ops.ScatterNdNonAliasingAdd(
            input=x,
            indices=indices,
            updates=updates,
        )
        out = out.numpy()
        return out

    class ThirdPartyImpl:
        """TensorFlow GPU golden。"""

        def __init__(self, x, indices, updates, **kwargs):
            self.x = tf.convert_to_tensor(x)
            self.indices = tf.convert_to_tensor(indices)
            self.updates = tf.convert_to_tensor(updates)

            if self.indices.shape.rank == 1:
                index_depth = self.indices.shape[-1]
                self.indices = tf.reshape(self.indices, (1, index_depth))
                self.updates = tf.reshape(
                    self.updates,
                    (1,) + tuple(self.x.shape[index_depth:]),
                )

        def __call__(self):
            if (
                tf.size(self.x).numpy() == 0
                or tf.size(self.indices).numpy() == 0
                or tf.size(self.updates).numpy() == 0
            ):
                return self.x
            if self.indices.shape[-1] > 5:
                return tf.convert_to_tensor(
                    _numpy_scatter_add(
                        self.x.numpy(), self.indices.numpy(), self.updates.numpy()
                    )
                )
            out = tf.raw_ops.ScatterNdNonAliasingAdd(
                input=self.x,
                indices=self.indices,
                updates=self.updates,
            )
            return out

    third_party = {"tf": ThirdPartyImpl}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "int32": {"standard": "binary_equal", "level": "L1"},
    }


class TfScatterNdNonAliasingAddSpec:
    """ScatterNdNonAliasingAdd 的 TensorFlow E2E 测试规范。"""

    @staticmethod
    def golden(
        input,
        indices,
        updates,
        bad_indices_policy="",
        name=None,
        **kwargs,
    ):
        del name, kwargs
        out = _scatter_nd_non_aliasing_add(
            input,
            indices,
            updates,
            bad_indices_policy,
        )
        return [out]

    class ThirdPartyImpl:
        """TensorFlow GPU 标杆。"""

        def __init__(
            self,
            input,
            indices,
            updates,
            bad_indices_policy="",
            name=None,
            device=None,
            **kwargs,
        ):
            del name, kwargs
            with tf.device(device or "/GPU:0"):
                self.input = tf.convert_to_tensor(input)
                self.indices = tf.convert_to_tensor(indices)
                self.updates = tf.convert_to_tensor(updates)
            self.bad_indices_policy = bad_indices_policy

        def __call__(self):
            return [
                _scatter_nd_non_aliasing_add(
                    self.input,
                    self.indices,
                    self.updates,
                    self.bad_indices_policy,
                )
            ]

    third_party = {
        "tf": ThirdPartyImpl,
    }

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "int32": {"standard": "binary_equal", "level": "L1"},
    }
