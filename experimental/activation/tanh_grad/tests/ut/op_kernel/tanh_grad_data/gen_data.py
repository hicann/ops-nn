# This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
# and is contributed to the CANN Open Software.

# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# All Rights Reserved.

# Authors (accounts):
# - Zhou Jianhua <@LePenseur>
# - Su Tonghua <@sutonghua>

# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import sys
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def tanh_grad_golden(y, dy):
    """Pure numpy implementation of tanh_grad: dx = dy * (1 - y * y)"""
    return dy.astype(np.float32) * (1.0 - y.astype(np.float32) * y.astype(np.float32))


def pack_as_bf16(arr_fp32):
    tmp = np.asarray(arr_fp32, dtype=np.float32)
    view_32 = tmp.view(np.uint32)
    bf16_u16 = (view_32 >> 16).astype(np.uint16)
    return bf16_u16


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": "bfloat16",
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)

    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    tmp_y = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_dy = np.random.choice(val_pool, size=size).reshape(shape)

    golden = tanh_grad_golden(tmp_y, tmp_dy)

    if d_type == "bfloat16":
        pack_as_bf16(tmp_y).tofile("bfloat16_y_t_tanh_grad.bin")
        pack_as_bf16(tmp_dy).tofile("bfloat16_dy_t_tanh_grad.bin")
        pack_as_bf16(golden).tofile("bfloat16_golden_t_tanh_grad.bin")
    else:
        tmp_y = tmp_y.astype(np_type)
        tmp_dy = tmp_dy.astype(np_type)

        tmp_y.tofile(f"{d_type}_y_t_tanh_grad.bin")
        tmp_dy.tofile(f"{d_type}_dy_t_tanh_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_t_tanh_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
