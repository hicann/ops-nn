#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import copy
import numpy as np
import re


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def compute_bias_correction(step, beta1, beta2):
    """Compute bias correction terms."""
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    bias_correction2_sqrt = np.sqrt(bias_correction2)
    return bias_correction1, bias_correction2_sqrt


def float32_to_bf16_bytes(arr):
    """Convert float32 array to bfloat16 bytes (round-to-nearest-even)."""
    u32 = arr.astype(np.float32).view(np.uint32)
    # Round to nearest even: add bias + (lsb of result)
    rounding_bias = np.where(u32 & 0x10000, 0x7FFF, 0x8000)
    u32 = u32 + rounding_bias
    bf16 = (u32 >> 16).astype(np.uint16)
    return bf16.tobytes()


def save_arr(arr, d_type, filename):
    """Save array to file with specified dtype."""
    if d_type == "bfloat16":
        with open(filename, "wb") as f:
            f.write(float32_to_bf16_bytes(arr.astype(np.float32)))
    else:
        np_dtype = np.float32 if d_type == "float32" else np.float16
        arr.astype(np_dtype).tofile(filename)


def gen_data_and_golden(shape_str, d_type="float32"):
    shape_list = parse_str_to_shape_list(shape_str)

    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.01
    eps = 1e-8
    amsgrad = 0
    maximize = 0
    use_grad_scale = 1
    grad_scale = 0.5
    step_count = 1.0  # step count as float

    for index, shape in enumerate(shape_list):
        params = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 10
        grads = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 10
        exp_avgs = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 0.1
        exp_avg_sqs = np.abs(np.random.rand(*shape) * 0.1).astype(np.float32)
        max_exp_avg_sqs = np.zeros_like(params)

        # Save inputs (state_steps 保存 step_count - 1，kernel 内部会 +1)
        save_arr(params, d_type, f"{d_type}_input_t_params_{index}.bin")
        save_arr(grads, d_type, f"{d_type}_input_t_grads_{index}.bin")
        save_arr(exp_avgs, d_type, f"{d_type}_input_t_exp_avgs_{index}.bin")
        save_arr(exp_avg_sqs, d_type, f"{d_type}_input_t_exp_avg_sqs_{index}.bin")
        save_arr(
            max_exp_avg_sqs, d_type, f"{d_type}_input_t_max_exp_avg_sqs_{index}.bin"
        )
        np.array([step_count - 1.0], dtype=np.float32).tofile(
            f"{d_type}_input_t_state_steps_{index}.bin"
        )

        # Compute reference (Adam in fp32 precision)
        p = params.astype(np.float32)
        g = grads.astype(np.float32)
        g2 = copy.deepcopy(grads).astype(np.float32)
        m = exp_avgs.astype(np.float32)
        v = exp_avg_sqs.astype(np.float32)
        mx = max_exp_avg_sqs.astype(np.float32)

        # Step 1: gradient scaling
        if use_grad_scale:
            g = g / grad_scale
            g2 = g2 / grad_scale

        # Step 2: maximize
        if maximize:
            g2 = -g2

        # Step 3: Adam weight decay
        if weight_decay != 0.0:
            g2 += p * weight_decay

        # Step 4: update first moment
        m = beta1 * m + (1.0 - beta1) * g2

        # Step 5: update second moment
        v = beta2 * v + (1.0 - beta2) * g2 * g2

        # Step 6: bias correction
        bias_correction1, bias_correction2_sqrt = compute_bias_correction(
            step_count, beta1, beta2
        )
        step_size = lr / bias_correction1

        # Step 7: denom
        if amsgrad:
            mx = np.maximum(mx, v)
            denom = np.sqrt(mx) / bias_correction2_sqrt + eps
        else:
            denom = np.sqrt(v) / bias_correction2_sqrt + eps

        # Step 8: update params
        p = p - step_size * m / denom

        # Save golden outputs
        save_arr(p, d_type, f"{d_type}_golden_t_params_ref_{index}.bin")
        save_arr(g, d_type, f"{d_type}_golden_t_grads_ref_{index}.bin")
        save_arr(m, d_type, f"{d_type}_golden_t_exp_avgs_ref_{index}.bin")
        save_arr(v, d_type, f"{d_type}_golden_t_exp_avg_sqs_ref_{index}.bin")
        if amsgrad:
            save_arr(mx, d_type, f"{d_type}_golden_t_max_exp_avg_sqs_ref_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
