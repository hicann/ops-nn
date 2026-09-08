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

"""MaxPoolV3Grad golden aligned with the common MaxPoolGrad SIMT kernel."""

__spec__ = {"max_pool_v3_grad": "MaxPoolV3GradKernelSpec"}

import ast
import numpy as np
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


def _as_bool(value):
    value = _parse(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return bool(value)


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

    if _as_bool(_attr(params, "global_pooling", False)):
        return input_h, input_w, 1, 1, 0, 0, output_h, output_w
    if mode == "SAME":
        pad_top = max((output_h - 1) * stride_h + kernel_h - input_h, 0) // 2
        pad_left = max((output_w - 1) * stride_w + kernel_w - input_w, 0) // 2
    elif mode == "VALID":
        pad_top, pad_left = 0, 0
    else:
        pad_top, _, pad_left, _ = _pads(_attr(params, "pads", [0, 0, 0, 0]))
    return kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, output_h, output_w


def _select_index(window, floating):
    batch, window_h, window_w, channels = window.shape
    index = np.full((batch, channels), -1, dtype=np.int64)
    value_dtype = np.float32 if floating else window.dtype
    max_value = np.zeros((batch, channels), dtype=value_dtype)
    spatial_index = 0
    for h_index in range(window_h):
        for w_index in range(window_w):
            value = np.asarray(window[:, h_index, w_index, :], dtype=value_dtype)
            update = (index < 0) | (value > max_value)
            if floating:
                update |= np.isnan(value)
            np.copyto(max_value, value, where=update)
            index[update] = spatial_index
            spatial_index += 1
    return index


def _tf_max_pool_grad(orig_input, orig_output, grad, pool_params):
    kernel_h, kernel_w, stride_h, stride_w, pad_top, pad_left, output_h, output_w = (
        pool_params
    )
    batch, input_h, input_w, channels = orig_input.shape.as_list()
    if 0 in (batch, input_h, input_w, channels, output_h, output_w):
        return tf.zeros_like(orig_input)

    input_data = orig_input.numpy()
    grad_data = grad.numpy()
    floating = orig_input.dtype.is_floating
    if floating:
        accumulator_dtype = np.float32
    elif orig_input.dtype.name.startswith("uint"):
        accumulator_dtype = np.uint64
    else:
        accumulator_dtype = np.int64
    result = np.zeros(input_data.shape, dtype=accumulator_dtype)
    grad_data = np.asarray(grad_data, dtype=accumulator_dtype)
    batch_index = np.arange(batch)[:, np.newaxis]
    channel_index = np.arange(channels)[np.newaxis, :]

    for output_h_index in range(output_h):
        raw_h_start = output_h_index * stride_h - pad_top
        h_start = max(raw_h_start, 0)
        h_end = min(raw_h_start + kernel_h, input_h)
        if h_start >= h_end:
            continue
        for output_w_index in range(output_w):
            raw_w_start = output_w_index * stride_w - pad_left
            w_start = max(raw_w_start, 0)
            w_end = min(raw_w_start + kernel_w, input_w)
            if w_start >= w_end:
                continue

            window = input_data[:, h_start:h_end, w_start:w_end, :]
            index = _select_index(window, floating)
            selected_h = h_start + index // (w_end - w_start)
            selected_w = w_start + index % (w_end - w_start)
            np.add.at(
                result,
                (batch_index, selected_h, selected_w, channel_index),
                grad_data[:, output_h_index, output_w_index, :],
            )
    return tf.convert_to_tensor(result, dtype=orig_input.dtype)


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

    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            self.params = kwargs

        def __call__(self, orig_input, orig_output, grad, **kwargs):
            return [_golden(orig_input, orig_output, grad, self.params)]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
        "int8": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
    }
