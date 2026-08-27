#!/usr/bin/env python3
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

"""MaxPoolV3Grad golden based on TensorFlow MaxPoolGrad."""

__spec__ = {"max_pool_v3_grad": "MaxPoolV3GradKernelSpec"}

import ast
import tensorflow as tf


_MISSING = object()


def _parse(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _attr(params, name, default=_MISSING):
    if name in params:
        return _parse(params[name])
    for key in (
        "attributes",
        "attrs",
        "other_compilation_params",
        "other_runtime_params",
    ):
        values = _parse(params.get(key))
        if isinstance(values, dict) and name in values:
            return _parse(values[name])
    if default is not _MISSING:
        return default
    raise RuntimeError("Required attribute [{}] is missing.".format(name))


def _tensor(value):
    try:
        return tf.convert_to_tensor(value)
    except (TypeError, ValueError, tf.errors.OpError):
        dtype = getattr(value, "dtype", None)
        if str(getattr(dtype, "name", dtype)).lower() == "bfloat16" and hasattr(
            value, "tolist"
        ):
            return tf.convert_to_tensor(value.tolist(), dtype=tf.bfloat16)
        raise


def _spatial(value, data_format):
    value = list(_parse(value))
    if len(value) == 1:
        return int(value[0]), int(value[0])
    if len(value) == 2:
        return int(value[0]), int(value[1])
    axes = (2, 3) if data_format == "NCHW" else (1, 2)
    return int(value[axes[0]]), int(value[axes[1]])


def _pads(value):
    value = list(_parse(value))
    if len(value) == 1:
        return int(value[0]), int(value[0]), int(value[0]), int(value[0])
    if len(value) == 2:
        return int(value[0]), int(value[0]), int(value[1]), int(value[1])
    return tuple(int(item) for item in value[:4])


def _pool_params(input_nhwc, output_nhwc, params, data_format):
    input_h, input_w = input_nhwc.shape.as_list()[1:3]
    output_h, output_w = output_nhwc.shape.as_list()[1:3]
    kernel_h, kernel_w = _spatial(_attr(params, "ksize"), data_format)
    stride_h, stride_w = _spatial(_attr(params, "strides"), data_format)
    mode = str(_attr(params, "padding_mode", "CALCULATED")).upper()

    if bool(_attr(params, "global_pooling", False)):
        return input_h, input_w, 1, 1, 0, 0, output_h, output_w
    if mode == "SAME":
        pad_top = max((output_h - 1) * stride_h + kernel_h - input_h, 0) // 2
        pad_left = max((output_w - 1) * stride_w + kernel_w - input_w, 0) // 2
    elif mode == "VALID":
        pad_top, pad_left = 0, 0
    else:
        pad_top, _, pad_left, _ = _pads(_attr(params, "pads", [0, 0, 0, 0]))
    return kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, output_h, output_w


def _encode(orig_input, orig_output):
    if orig_input.dtype.is_floating:
        input_data, output_data = (
            tf.cast(orig_input, tf.float32),
            tf.cast(orig_output, tf.float32),
        )
        input_valid, output_valid = (
            ~tf.math.is_nan(input_data),
            ~tf.math.is_nan(output_data),
        )
        has_special = tf.reduce_any(
            ~input_valid | (tf.math.is_inf(input_data) & (input_data < 0))
        )
        if not bool(has_special.numpy()):
            output_data = tf.where(
                output_valid, output_data, tf.zeros_like(output_data)
            )
            return (
                input_data,
                output_data,
                input_valid,
                output_valid,
                tf.constant(float("-inf")),
                True,
            )
        valid_values, compute_dtype = (
            tf.boolean_mask(input_data, input_valid),
            tf.float32,
        )
    else:
        input_data, output_data = (
            tf.cast(orig_input, tf.int64),
            tf.cast(orig_output, tf.int64),
        )
        input_valid, output_valid = (
            tf.ones_like(input_data, tf.bool),
            tf.ones_like(output_data, tf.bool),
        )
        min_value = int(tf.reduce_min(input_data).numpy())
        if min_value > -9223372036854775808:
            return (
                input_data,
                output_data,
                input_valid,
                output_valid,
                tf.constant(min_value - 1, tf.int64),
                True,
            )
        valid_values, compute_dtype = tf.reshape(input_data, [-1]), tf.int64

    if int(tf.size(valid_values).numpy()) == 0:
        return (
            input_data,
            output_data,
            input_valid,
            output_valid,
            tf.cast(0, compute_dtype),
            False,
        )

    unique_values = tf.unique(tf.sort(tf.reshape(valid_values, [-1]))).y

    def rank(data, valid):
        data = tf.where(valid, data, tf.zeros_like(data))
        value = (
            tf.searchsorted(
                unique_values, tf.reshape(data, [-1]), side="left", out_type=tf.int64
            )
            + 1
        )
        value = tf.cast(tf.reshape(value, tf.shape(data)), compute_dtype)
        return tf.where(valid, value, tf.zeros_like(value))

    return (
        rank(input_data, input_valid),
        rank(output_data, output_valid),
        input_valid,
        output_valid,
        tf.cast(0, compute_dtype),
        True,
    )


def _tf_max_pool_grad(orig_input, orig_output, grad, pool_params):
    kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, output_h, output_w = (
        pool_params
    )
    batch, input_h, input_w, channels = orig_input.shape.as_list()
    if 0 in (batch, input_h, input_w, channels, output_h, output_w):
        return tf.zeros_like(orig_input)

    target_h, target_w = (
        (output_h - 1) * stride_h + kernel_h,
        (output_w - 1) * stride_w + kernel_w,
    )
    if pad_top >= target_h or pad_left >= target_w:
        return tf.zeros_like(orig_input)
    keep_h, keep_w = min(input_h, target_h - pad_top), min(input_w, target_w - pad_left)
    if keep_h <= 0 or keep_w <= 0:
        return tf.zeros_like(orig_input)

    input_data, _, input_valid, _, pad_value, has_valid = _encode(
        orig_input, orig_output
    )
    if not has_valid:
        return tf.zeros_like(orig_input)

    size = [batch, keep_h, keep_w, channels]
    input_data = tf.slice(input_data, [0, 0, 0, 0], size)
    input_valid = tf.slice(input_valid, [0, 0, 0, 0], size)
    pad_bottom, pad_right = target_h - pad_top - keep_h, target_w - pad_left - keep_w
    paddings = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]
    input_data = tf.pad(input_data, paddings, constant_values=pad_value)
    input_valid = tf.pad(input_valid, paddings, constant_values=False)

    valid_window = (
        tf.nn.max_pool2d(
            tf.cast(input_valid, tf.float32),
            [1, kernel_h, kernel_w, 1],
            [1, stride_h, stride_w, 1],
            "VALID",
            data_format="NHWC",
        )
        > 0
    )
    output_data = tf.nn.max_pool2d(
        input_data,
        [1, kernel_h, kernel_w, 1],
        [1, stride_h, stride_w, 1],
        "VALID",
        data_format="NHWC",
    )
    grad_data = tf.cast(grad, input_data.dtype)
    grad_data = tf.where(valid_window, grad_data, tf.zeros_like(grad_data))
    padded_grad = tf.raw_ops.MaxPoolGrad(
        orig_input=input_data,
        orig_output=output_data,
        grad=grad_data,
        ksize=[1, kernel_h, kernel_w, 1],
        strides=[1, stride_h, stride_w, 1],
        padding="VALID",
        explicit_paddings=[],
        data_format="NHWC",
    )
    result = tf.slice(padded_grad, [0, pad_top, pad_left, 0], size)
    result = tf.pad(
        result, [[0, 0], [0, input_h - keep_h], [0, input_w - keep_w], [0, 0]]
    )
    return tf.cast(result, orig_input.dtype)


def _golden(orig_input, orig_output, grad, params):
    input_tensor, output_tensor, grad_tensor = (
        _tensor(orig_input),
        _tensor(orig_output),
        _tensor(grad),
    )
    data_format = str(_attr(params, "data_format", "NCHW")).upper()
    if data_format == "NCHW":
        input_nhwc, output_nhwc, grad_nhwc = (
            tf.transpose(input_tensor, [0, 2, 3, 1]),
            tf.transpose(output_tensor, [0, 2, 3, 1]),
            tf.transpose(grad_tensor, [0, 2, 3, 1]),
        )
    else:
        input_nhwc, output_nhwc, grad_nhwc = input_tensor, output_tensor, grad_tensor
    if 0 in input_nhwc.shape.as_list():
        return tf.zeros_like(input_tensor).numpy()

    pool_params = _pool_params(input_nhwc, output_nhwc, params, data_format)
    result = _tf_max_pool_grad(input_nhwc, output_nhwc, grad_nhwc, pool_params)
    if data_format == "NCHW":
        result = tf.transpose(result, [0, 3, 1, 2])
    return tf.cast(result, input_tensor.dtype).numpy()


class MaxPoolV3GradKernelSpec:
    @staticmethod
    def golden(orig_input, orig_output, grad, **kwargs):
        return [_golden(orig_input, orig_output, grad, kwargs)]

    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "float32": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
