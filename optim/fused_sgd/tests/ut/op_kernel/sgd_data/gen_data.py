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
import numpy as np
import re


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)

    weight_decay = 0.01
    momentum = 0.0
    lr = 0.001
    dampening = 0.0
    nesterov = 0
    maximize = 0
    is_first_step = 1
    use_grad_scale = 1
    use_momentum = 0
    grad_scale = 0.5

    for index, shape in enumerate(shape_list):
        params = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 100
        grads = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 100
        momentum_buf = (np.random.rand(*shape) * 2 - 1).astype(np.float32) * 100

        grads_inv = grads.copy()
        if use_grad_scale:
            inv_grad_scale = 1.0 / grad_scale
            grads_inv = grads_inv * inv_grad_scale

        grads_ref = grads_inv.copy()

        if maximize:
            grads_inv = -grads_inv

        if weight_decay != 0.0:
            grads_inv = grads_inv + weight_decay * params

        if use_momentum:
            if is_first_step:
                momentum_buf_out = grads_inv.copy()
            else:
                momentum_buf_out = momentum * momentum_buf + (1.0 - dampening) * grads_inv
            if nesterov:
                grads_inv = grads_inv + momentum * momentum_buf_out
            else:
                grads_inv = momentum_buf_out.copy()
        else:
            momentum_buf_out = momentum_buf.copy()

        params_ref = params - lr * grads_inv

        params.astype(np_type).tofile(f"{d_type}_input_t_params_{index}.bin")
        grads.astype(np_type).tofile(f"{d_type}_input_t_grads_{index}.bin")
        momentum_buf.astype(np_type).tofile(f"{d_type}_input_t_momentum_{index}.bin")
        params.astype(np_type).tofile(f"{d_type}_input_t_params_ref_{index}.bin")
        grads.astype(np_type).tofile(f"{d_type}_input_t_grads_ref_{index}.bin")
        momentum_buf.astype(np_type).tofile(f"{d_type}_input_t_momentum_ref_{index}.bin")

        params_ref.astype(np_type).tofile(f"{d_type}_golden_t_params_ref_{index}.bin")
        grads_ref.astype(np_type).tofile(f"{d_type}_golden_t_grads_ref_{index}.bin")
        momentum_buf_out.astype(np_type).tofile(f"{d_type}_golden_t_momentum_ref_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])