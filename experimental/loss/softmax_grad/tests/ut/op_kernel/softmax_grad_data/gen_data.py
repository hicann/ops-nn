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
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def softmax_grad_golden(y, dy):
    """纯 numpy 实现 softmax_grad golden 计算：
    dx[i,j] = y[i,j] * (dy[i,j] - sum_j(dy[i,j] * y[i,j]))
    """
    dot = np.sum(dy * y, axis=1, keepdims=True)
    return y * (dy - dot)


def pack_as_bf16(arr_fp32):
    """将 float32 数组转换为 bfloat16 的 uint16 表示（截断低16位尾数）。"""
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

    # 生成随机测试数据（含边界值、特殊值）
    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    tmp_y = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_dy = np.random.choice(val_pool, size=size).reshape(shape)

    if d_type == "bfloat16":
        # golden 在 float32 下计算
        golden = softmax_grad_golden(
            tmp_y.astype(np.float32), tmp_dy.astype(np.float32)
        )
        pack_as_bf16(tmp_y).tofile(
            "bfloat16_y_t_softmax_grad.bin"
        )
        pack_as_bf16(tmp_dy).tofile(
            "bfloat16_dy_t_softmax_grad.bin"
        )
        pack_as_bf16(golden).tofile(
            "bfloat16_golden_dx_t_softmax_grad.bin"
        )
    else:
        tmp_y = tmp_y.astype(np_type)
        tmp_dy = tmp_dy.astype(np_type)
        golden = softmax_grad_golden(
            tmp_y.astype(np.float32), tmp_dy.astype(np.float32)
        )

        tmp_y.tofile(f"{d_type}_y_t_softmax_grad.bin")
        tmp_dy.tofile(f"{d_type}_dy_t_softmax_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_dx_t_softmax_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
