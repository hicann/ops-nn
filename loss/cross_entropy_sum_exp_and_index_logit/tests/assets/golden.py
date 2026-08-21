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

import numpy as np

__golden__ = {
    "kernel": {
        "cross_entropy_sum_exp_and_index_logit": "cross_entropy_sum_exp_and_index_logit_golden"
    }
}


def cross_entropy_sum_exp_and_index_logit_golden(
    vocab_parallel_logits,
    target,
    global_logits_max,
    vocab_start_index=0,
    vocab_end_index=0,
    **kwargs,
):
    """
    Golden function for cross_entropy_sum_exp_and_index_logit.
    All the parameters (names and order) follow @cross_entropy_sum_exp_and_index_logit_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    target_shape = target.shape
    logits_shape = vocab_parallel_logits.shape

    logits = vocab_parallel_logits.astype(np.float32)
    tgt = target.astype(np.int64)
    gmax = global_logits_max.astype(np.float32)

    s = int(vocab_start_index)
    e = int(vocab_end_index)

    v_local = logits.shape[-1]
    n = logits.size // v_local
    if e == 0:
        e = v_local

    logits_2d = logits.reshape(n, v_local)
    target_1d = tgt.reshape(n)
    gmax_1d = gmax.reshape(n)

    # Step 1: shifted = logits - gmax
    shifted = logits_2d - gmax_1d[:, None]

    # Step 2: target_mask = (target < s or target >= e) ? 1 : 0
    target_mask_1d = ((target_1d < s) | (target_1d >= e)).astype(np.int32)

    # Step 3: target_offset = mask ? 0 : target - s
    target_offset_1d = np.where(target_mask_1d.astype(bool), 0, target_1d - s).astype(
        np.int32
    )

    # Step 4: predicted_logits gather from ORIGINAL logits
    gathered = logits_2d[np.arange(n), target_offset_1d.astype(np.int64)]
    predicted_1d = (gathered - gmax_1d) * (1.0 - target_mask_1d.astype(np.float32))

    # Step 5: exp + sum
    exp_logits = np.exp(shifted)
    sum_exp_1d = exp_logits.sum(axis=-1)

    return [
        predicted_1d.reshape(target_shape).astype(np.float32),
        sum_exp_1d.reshape(target_shape).astype(np.float32),
        exp_logits.reshape(logits_shape).astype(np.float32),
        target_offset_1d.reshape(target_shape).astype(np.int32),
        target_mask_1d.reshape(target_shape).astype(np.int32),
    ]
