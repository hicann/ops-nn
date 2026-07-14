# This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
# and is contributed to the CANN Open Software.

# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# All Rights Reserved.

# Authors (accounts):
# - Pei Haobo<@xiaopei-1>
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
    shape_str = shape_str.strip("(").strip(")").strip()
    shape_list = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
    return np.array(shape_list)


def relu_grad_v3_golden(x, y):
    """纯 numpy 实现 relu_grad_v3 golden 计算：
    z = (x > 0) ? y : 0
    """
    return np.where(x > 0, y, 0.0)


def pack_as_bf16(arr_fp32):
    """将 float32 数组转换为 bfloat16 的 uint16 表示（截断低16位尾数）。"""
    tmp = np.asarray(arr_fp32, dtype=np.float32)
    view_32 = tmp.view(np.uint32)
    bf16_u16 = (view_32 >> 16).astype(np.uint16)
    return bf16_u16


def gen_data_and_golden(shape_str, d_type="float32", y_shape_str=None):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": "bfloat16",
        "int32": np.int32,
        "uint8": np.uint8,
        "int8": np.int8,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    y_shape = parse_str_to_shape_list(y_shape_str) if y_shape_str is not None else shape
    size = np.prod(shape)
    y_size = np.prod(y_shape)

    # 生成随机测试数据（含边界值、特殊值）
    if d_type in ("int32", "uint8", "int8"):
        if d_type == "uint8":
            val_pool = [0, 1, 2, 7, 127, 255]
        elif d_type == "int8":
            val_pool = [-8, -2, -1, 0, 1, 2, 7]
        else:
            val_pool = [-1024, -2, -1, 0, 1, 2, 1024]
    else:
        val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 65504.0]
    tmp_x = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_y = np.random.choice(val_pool, size=y_size).reshape(y_shape)

    if d_type == "bfloat16":
        # golden 在 float32 下计算
        golden = relu_grad_v3_golden(tmp_x.astype(np.float32), tmp_y.astype(np.float32))
        pack_as_bf16(tmp_x).tofile("bfloat16_x_t_relu_grad_v3.bin")
        pack_as_bf16(tmp_y).tofile("bfloat16_y_t_relu_grad_v3.bin")
        pack_as_bf16(golden).tofile("bfloat16_golden_z_t_relu_grad_v3.bin")
    else:
        tmp_x = tmp_x.astype(np_type)
        tmp_y = tmp_y.astype(np_type)
        golden = relu_grad_v3_golden(tmp_x.astype(np.float32), tmp_y.astype(np.float32))

        tmp_x.tofile(f"{d_type}_x_t_relu_grad_v3.bin")
        tmp_y.tofile(f"{d_type}_y_t_relu_grad_v3.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_z_t_relu_grad_v3.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type> [y_shape_str]")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    y_shape_str = sys.argv[3] if len(sys.argv) > 3 else None
    gen_data_and_golden(sys.argv[1], sys.argv[2], y_shape_str)
