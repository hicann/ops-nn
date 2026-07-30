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

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

__golden__ = {"kernel": {"dilation2_d": "dilation2_d_golden"}}


def dilation2_d_golden(
    x,
    filter,
    *,
    strides=None,
    rates=None,
    padding_mode="SAME",
    pads=None,
    ceil_mode=False,
    data_format="NHWC",
    **kwargs,
):
    """
    Golden function for dilation2_d.
    All the parameters (names and order) follow @dilation2_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Uses TensorFlow's tf.nn.dilation2d as the reference computation engine, but
    manually computes padding and output size per the SE document formulas to
    ensure consistency with the kernel implementation.

    Morphological dilation 2D operation.
    y[b, oh, ow, c] = max_{fh, fw} (x[b, oh*stride_h + fh*rate_h, ow*stride_w + fw*rate_w, c] + filter[fh, fw, c])

    Args:
        x: numpy array, shape (N, H, W, C) for NHWC or (N, C, H, W) for NCHW
        filter: numpy array, shape (fH, fW, C)
        strides: list of 4 ints, [stride_n, stride_h, stride_w, stride_c] (NHWC) or
                 [stride_n, stride_c, stride_h, stride_w] (NCHW)
        rates: list of 4 ints, [rate_n, rate_h, rate_w, rate_c] (NHWC) or
               [rate_n, rate_c, rate_h, rate_w] (NCHW)
        padding_mode: str, "SAME", "VALID", or "CALCULATED"
        pads: list of 4 ints, explicit padding [top, left, bottom, right] (CALCULATED mode)
        ceil_mode: bool, use ceil or floor for output size (CALCULATED mode only)
        data_format: str, "NHWC" or "NCHW"
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray) in the same data_format as input.
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    if rates is None:
        rates = [1, 1, 1, 1]

    orig_dtype = x.dtype

    # ------------------------------------------------------------------
    # Step 1: NCHW -> NHWC conversion
    #         tf.nn.dilation2d only supports NHWC data_format.
    # ------------------------------------------------------------------
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        filter = np.transpose(filter, (1, 2, 0))
        # NCHW strides/rates: [N, C, H, W] -> extract H, W from index 2, 3
        stride_h = strides[2]
        stride_w = strides[3]
        rate_h = rates[2]
        rate_w = rates[3]
    else:
        # NHWC strides/rates: [N, H, W, C] -> extract H, W from index 1, 2
        stride_h = strides[1]
        stride_w = strides[2]
        rate_h = rates[1]
        rate_w = rates[2]

    N, H, W, C = x.shape
    fH, fW, fC = filter.shape

    # Effective filter size (with rates)
    filter_h_eff = fH + (fH - 1) * (rate_h - 1)
    filter_w_eff = fW + (fW - 1) * (rate_w - 1)

    # tf.nn.dilation2d expects NHWC-order strides and dilations
    tf_strides = [1, stride_h, stride_w, 1]
    tf_dilations = [1, rate_h, rate_w, 1]

    # ------------------------------------------------------------------
    # Step 2: Determine compute dtype
    #         tf.nn.dilation2d natively supports float16/32/64, int8/16/32/64,
    #         uint8/16/32/64, bfloat16. For integer types, use float64 for
    #         computation to avoid overflow in x+filter addition, then cast back.
    # ------------------------------------------------------------------
    compute_dtype = np.float64

    # ------------------------------------------------------------------
    # Step 3: Manually compute padding and output size per SE doc formula.
    #
    # WHY manual padding instead of tf.nn.dilation2d's built-in SAME/VALID:
    #
    # 1. TF SAME padding distribution differs from SE doc when rates > 1.
    #    SE doc: pad_top = pad_h // 2, pad_bottom = pad_h - pad_top
    #    TF internally may distribute padding differently for dilated filters.
    #
    # 2. TF VALID output size formula differs from SE doc:
    #    SE doc:  oH = ceil((H - filter_eff + 1) / stride)
    #    TF:      oH = ceil((H - filter_eff) / stride) + 1
    #    When stride > 1 and (H - filter_eff) is not divisible by stride,
    #    TF produces one extra row/col compared to SE doc.
    #
    # SOLUTION: Always compute padding and output size manually (per SE doc),
    # then pad the input explicitly with -inf (so padded positions don't
    # affect the max), call tf.nn.dilation2d with VALID, and finally crop
    # the output to the SE-computed oH x oW.
    #
    # Window positions are consistent because both SE doc and TF use:
    #   ih = oh * stride_h + fh * rate_h - pad_top
    # So the first oH rows / oW cols of TF's output are the correct results.
    # ------------------------------------------------------------------
    if padding_mode == "SAME":
        oH = int(np.ceil(H / stride_h))
        oW = int(np.ceil(W / stride_w))
        pad_h = max((oH - 1) * stride_h + filter_h_eff - H, 0)
        pad_w = max((oW - 1) * stride_w + filter_w_eff - W, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    elif padding_mode == "VALID":
        oH = int(np.ceil((H - filter_h_eff + 1) / stride_h))
        oW = int(np.ceil((W - filter_w_eff + 1) / stride_w))
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
    else:
        # CALCULATED mode: pads = [H_top, H_bottom, W_left, W_right] (canndev compatible)
        pad_top = pads[0] if pads is not None and len(pads) > 0 else 0
        pad_bottom = pads[1] if pads is not None and len(pads) > 1 else 0
        pad_left = pads[2] if pads is not None and len(pads) > 2 else 0
        pad_right = pads[3] if pads is not None and len(pads) > 3 else 0
        H_padded = H + pad_top + pad_bottom
        W_padded = W + pad_left + pad_right
        if ceil_mode:
            oH = int(np.ceil((H_padded - filter_h_eff) / stride_h)) + 1
            oW = int(np.ceil((W_padded - filter_w_eff) / stride_w)) + 1
        else:
            oH = int(np.floor((H_padded - filter_h_eff) / stride_h)) + 1
            oW = int(np.floor((W_padded - filter_w_eff) / stride_w)) + 1

    oH = max(oH, 0)
    oW = max(oW, 0)

    has_padding = pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0

    # ------------------------------------------------------------------
    # Step 4: Handle empty output (oH=0 or oW=0) without calling TF
    # ------------------------------------------------------------------
    if oH == 0 or oW == 0:
        y = np.empty((N, oH, oW, C), dtype=compute_dtype)
    else:
        # Convert numpy arrays to TensorFlow tensors
        x_tf = tf.constant(x.astype(compute_dtype))
        filter_tf = tf.constant(filter.astype(compute_dtype))

        # Explicitly pad input with -inf so padded positions do not affect max
        if has_padding:
            paddings = tf.constant(
                [
                    [0, 0],  # N dimension: no padding
                    [pad_top, pad_bottom],  # H dimension
                    [pad_left, pad_right],  # W dimension
                    [0, 0],  # C dimension: no padding
                ],
                dtype=tf.int32,
            )
            x_padded = tf.pad(
                x_tf, paddings, mode="CONSTANT", constant_values=float("-inf")
            )
        else:
            x_padded = x_tf

        # ----------------------------------------------------------------
        # Step 5: Execute dilation via tf.nn.dilation2d with VALID padding
        #         (always VALID, since we handle padding manually)
        # ----------------------------------------------------------------
        y_tf = tf.nn.dilation2d(
            input=x_padded,
            filters=filter_tf,
            strides=tf_strides,
            padding="VALID",
            data_format="NHWC",
            dilations=tf_dilations,
        )
        y = y_tf.numpy()

        # ----------------------------------------------------------------
        # Step 6: Align output to canndev-computed oH x oW.
        #
        # tf.nn.dilation2d VALID uses oH_tf = ceil((H_padded - filter_eff + 1) / stride),
        # while canndev CALCULATED uses oH = ceil((H_padded - filter_eff) / stride) + 1.
        # When (H_padded - filter_eff) % stride != 0, canndev produces one more
        # row/col than TF. Window positions are aligned (ih = oh*stride + fh*rate
        # in padded coords), so the first min(oH, oH_tf) rows/cols are identical.
        # Extra rows/cols beyond TF's output are computed manually (same as kernel:
        # out-of-bounds positions are skipped, i.e. treated as -inf).
        # ----------------------------------------------------------------
        tf_oH = y.shape[1]
        tf_oW = y.shape[2]
        if tf_oH < oH or tf_oW < oW:
            x_np = x_padded.numpy()
            f_np = filter.astype(compute_dtype)
            full_y = np.full((N, oH, oW, C), -np.inf, dtype=compute_dtype)
            full_y[:, : min(tf_oH, oH), : min(tf_oW, oW), :] = y[
                :, : min(tf_oH, oH), : min(tf_oW, oW), :
            ]
            for b in range(N):
                for oh in range(oH):
                    for ow in range(oW):
                        if oh < tf_oH and ow < tf_oW:
                            continue
                        for fh_idx in range(fH):
                            for fw_idx in range(fW):
                                ih = oh * stride_h + fh_idx * rate_h
                                iw = ow * stride_w + fw_idx * rate_w
                                if 0 <= ih < x_np.shape[1] and 0 <= iw < x_np.shape[2]:
                                    val = x_np[b, ih, iw, :] + f_np[fh_idx, fw_idx, :]
                                    full_y[b, oh, ow, :] = np.maximum(
                                        full_y[b, oh, ow, :], val
                                    )
            y = full_y
        else:
            y = y[:, :oH, :oW, :]

        # ----------------------------------------------------------------
        # Step 7: Restore -inf for all-padded windows (only when padding
        #         was applied).
        #
        # tf.nn.dilation2d initializes its internal max accumulator to
        # finfo(compute_dtype).min. When ALL window positions fall on padded
        # (-inf) input cells, -inf + filter = -inf is still less than the
        # accumulator initial value (finfo.min), so the output retains
        # finfo.min instead of the mathematically correct -inf.
        #
        # We replace any output value <= finfo.min with -inf to restore the
        # correct semantics: out-of-bounds positions yield -inf.
        # ----------------------------------------------------------------
        if has_padding:
            compute_finfo_min = np.finfo(compute_dtype).min
            y = np.where(y <= compute_finfo_min, -np.inf, y)

    # ------------------------------------------------------------------
    # Step 8: Convert back to original dtype
    # ------------------------------------------------------------------
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        y = np.clip(np.round(y), info.min, info.max).astype(orig_dtype)
    else:
        y = y.astype(orig_dtype)

    # ------------------------------------------------------------------
    # Step 9: NHWC -> NCHW conversion (if original data_format was NCHW)
    # ------------------------------------------------------------------
    if data_format == "NCHW":
        y = np.transpose(y, (0, 3, 1, 2))

    return y
