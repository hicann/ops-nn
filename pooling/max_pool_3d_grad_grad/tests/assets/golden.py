# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""References for Kernel/GEIR and TensorFlow framework E2E test paths.

Kernel and GEIR pass ``numpy.ndarray`` inputs to ``golden`` and share the
``max_pool_3d_grad_grad`` registration key.  The optional third-party provider
receives TensorFlow tensors and calls the same TensorFlow RawOp.

TensorFlow 2.21 CPU has no FP16 kernel for ``MaxPool3DGradGrad``.  This spec
therefore computes the TensorFlow reference in FP32 and casts the result back
to FP16.  The reference keeps TensorFlow semantics: no equality match returns
zero, and SAME padding compares only valid input positions.
"""

from typing import ClassVar

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context as tf_context

__spec__ = {
    "max_pool_3d_grad_grad": "MaxPool3DGradGradKernelGeirSpec",
    "MaxPool3DGradGrad": "MaxPool3DGradGradKernelGeirSpec",
    "tf.raw_ops.MaxPool3DGradGrad": "MaxPool3DGradGradE2ETestSpec",
}


# Test-case names carrying this marker keep the caller-provided orig_y instead of the
# forward result, so the "no equality match or NaN target -> FP16 zero" branch stays
# reachable; every other case receives a forward-consistent orig_y (match branch).
NO_MATCH_CASE_MARKERS = ("nomatch", "no_match")


def _is_no_match_case(testcase_name):
    name = str(testcase_name).lower()
    return any(marker in name for marker in NO_MATCH_CASE_MARKERS)


def _canonical_5d(values, name):
    """Convert a scalar/3D/5D pooling attribute to TensorFlow NDHWC form."""
    values = tuple(int(value) for value in values)
    if len(values) == 1:
        spatial = values * 3
    elif len(values) == 3:
        spatial = values
    elif len(values) == 5:
        if values[0] != 1 or values[4] != 1:
            raise ValueError(f"{name} must not pool N/C axes")
        spatial = values[1:4]
    else:
        raise ValueError(f"{name} length must be 1, 3 or 5")
    if any(value <= 0 for value in spatial):
        raise ValueError(f"{name} spatial values must be positive")
    return [1, spatial[0], spatial[1], spatial[2], 1]


def _padding_from_pads(pads):
    pads = tuple(int(value) for value in pads)
    if len(pads) != 6 or any(value < 0 for value in pads):
        raise ValueError("pads must contain 6 non-negative integers")
    return "VALID" if all(value == 0 for value in pads) else "SAME"


def _tensorflow_max_pool_3d_grad_grad(
    orig_x, orig_y, grads, *, ksize, strides, pads, data_format="NDHWC"
):
    """Run the same-name TensorFlow RawOp on CPU and return the FP16 operator result.

    float32 input is accepted because TTK promotes float16 inputs to float32 for the
    reference when the registered tolerance standard is ``cross_check``; promotion is
    lossless for a float16 test case and the operator output contract stays float16.
    """
    if data_format != "NDHWC":
        raise ValueError("only logical NDHWC is supported")

    x_tensor = tf.convert_to_tensor(orig_x)
    y_tensor = tf.convert_to_tensor(orig_y)
    grads_tensor = tf.convert_to_tensor(grads)
    if x_tensor.dtype != y_tensor.dtype or x_tensor.dtype != grads_tensor.dtype:
        raise TypeError("orig_x, orig_y and grads must have the same dtype")
    if x_tensor.dtype not in (tf.float16, tf.float32):
        raise TypeError(
            f"only float16 (or float32-promoted float16) inputs are supported, got {x_tensor.dtype.name}"
        )
    if (
        x_tensor.shape.rank != 5
        or y_tensor.shape.rank != 5
        or grads_tensor.shape.rank != 5
    ):
        raise ValueError("orig_x, orig_y and grads must be rank 5")
    if x_tensor.shape != grads_tensor.shape:
        raise ValueError("orig_x and grads must have the same shape")

    output = tf.raw_ops.MaxPool3DGradGrad(
        orig_input=tf.cast(x_tensor, tf.float32),
        orig_output=tf.cast(y_tensor, tf.float32),
        grad=tf.cast(grads_tensor, tf.float32),
        ksize=_canonical_5d(ksize, "ksize"),
        strides=_canonical_5d(strides, "strides"),
        padding=_padding_from_pads(pads),
        data_format=data_format,
    )
    return tf.cast(output, tf.float16)


class MaxPool3DGradGradKernelGeirSpec:
    """Kernel/GEIR TestSpec using TensorFlow MaxPool3DGradGrad on CPU."""

    def customize_inputs(
        orig_x,
        orig_y,
        grads,
        *,
        ksize,
        strides,
        pads,
        data_format="NDHWC",
        **kwargs,
    ):
        """Fill in a forward-consistent orig_y unless the case asks to keep its own.

        Shapes are always checked against the forward result. Values are replaced by the
        forward output only for cases that are not marked as no-match cases, otherwise the
        supplied orig_y is preserved so the zero-output branch remains testable.
        """
        testcase_name = kwargs.pop("testcase_name", "")
        del kwargs
        if data_format != "NDHWC":
            raise ValueError("only logical NDHWC is supported")
        x_array = np.ascontiguousarray(orig_x, dtype=np.float16)
        grads_array = np.ascontiguousarray(grads, dtype=np.float16)
        pooled = tf.nn.max_pool3d(
            tf.cast(tf.convert_to_tensor(x_array), tf.float32),
            ksize=_canonical_5d(ksize, "ksize"),
            strides=_canonical_5d(strides, "strides"),
            padding=_padding_from_pads(pads),
            data_format=data_format,
        )
        pooled_array = np.ascontiguousarray(tf.cast(pooled, tf.float16).numpy())
        if pooled_array.shape != orig_y.shape:
            raise ValueError(
                f"TensorFlow forward shape {pooled_array.shape} does not match "
                f"orig_y shape {orig_y.shape}"
            )
        if _is_no_match_case(testcase_name):
            return x_array, np.ascontiguousarray(orig_y, dtype=np.float16), grads_array
        return x_array, pooled_array, grads_array

    def golden(
        orig_x,
        orig_y,
        grads,
        *,
        ksize,
        strides,
        pads,
        data_format="NDHWC",
        **kwargs,
    ):
        """Return a one-slot list containing the NumPy Golden output."""
        del kwargs
        result = _tensorflow_max_pool_3d_grad_grad(
            orig_x,
            orig_y,
            grads,
            ksize=ksize,
            strides=strides,
            pads=pads,
            data_format=data_format,
        )
        return [np.ascontiguousarray(result.numpy())]

    class TensorFlowThirdPartyImpl:
        """TensorFlow provider implementation used by optional cross-checks."""

        def __init__(
            self,
            *,
            ksize,
            strides,
            pads,
            data_format="NDHWC",
            **kwargs,
        ):
            del kwargs
            self.ksize = ksize
            self.strides = strides
            self.pads = pads
            self.data_format = data_format

        def __call__(self, orig_x, orig_y, grads, **kwargs):
            del kwargs
            return [
                _tensorflow_max_pool_3d_grad_grad(
                    orig_x,
                    orig_y,
                    grads,
                    ksize=self.ksize,
                    strides=self.strides,
                    pads=self.pads,
                    data_format=self.data_format,
                )
            ]

    third_party: ClassVar[dict] = {"tf": TensorFlowThirdPartyImpl}
    tolerance: ClassVar[dict] = {"float16": {"standard": "cross_check", "level": "L1"}}


class MaxPool3DGradGradE2ETestSpec:
    """TF API parameters are distinct from the GEIR/kernel pads interface."""

    @staticmethod
    def golden(
        orig_input,
        orig_output,
        grad,
        ksize,
        strides,
        padding,
        data_format="NDHWC",
        name=None,
        **kwargs,
    ):
        del name, kwargs
        if padding not in ("SAME", "VALID"):
            raise ValueError("padding must be SAME or VALID")
        pads = [0, 0, 0, 0, 0, 0] if padding == "VALID" else [1, 1, 1, 1, 1, 1]
        # npu_device.as_default() overrides tf.device; use the eager CPU scope.
        with tf_context.device("/CPU:0"):
            inputs = [
                tf.convert_to_tensor(
                    value.numpy() if hasattr(value, "numpy") else value
                )
                for value in (orig_input, orig_output, grad)
            ]
            result = _tensorflow_max_pool_3d_grad_grad(
                *inputs,
                ksize=ksize,
                strides=strides,
                pads=pads,
                data_format=data_format,
            )
        return [result]

    third_party: ClassVar[dict] = {"tf": "tf.raw_ops.MaxPool3DGradGrad"}
    tolerance: ClassVar[dict] = {"float16": {"standard": "cross_check", "level": "L1"}}
