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


# 256B 对齐参数（与 tiling.cpp 一致）
BLOCK_SIZE = 256


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def align_to_block(num_elements, type_size):
    """将元素数向上对齐到 256B block 边界"""
    input_length = num_elements * type_size
    aligned_length = ((input_length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    return aligned_length // type_size


def max_pooling_grad_golden(dy, x, y):
    """纯 numpy 实现 max_pooling_grad golden:
    dx[i] = (x[i] == y[i]) ? dy[i] : 0
    使用 np.isclose 做浮点安全比较
    """
    mask = np.isclose(x.astype(np.float64), y.astype(np.float64), atol=1e-6)
    dx = np.where(mask, dy, np.float32(0))
    return dx.astype(np.float32)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    type_size_dict = {
        "float32": 4,
        "float16": 2,
    }
    np_type = d_type_dict[d_type]
    type_size = type_size_dict[d_type]

    shape = parse_str_to_shape_list(shape_str)
    original_size = int(np.prod(shape))

    # kernel 按 256B 对齐分配内存，gen_data 也按对齐大小生成
    aligned_size = align_to_block(original_size, type_size)

    # 生成测试数据（前 original_size 元素），含特殊值
    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    tmp_dy = np.zeros(aligned_size, dtype=np.float32)
    tmp_x = np.zeros(aligned_size, dtype=np.float32)
    tmp_y = np.zeros(aligned_size, dtype=np.float32)

    # 填充前 original_size 元素为随机测试值
    tmp_dy[:original_size] = np.random.choice(val_pool, size=original_size)
    tmp_x[:original_size] = np.random.choice(val_pool, size=original_size)

    # y = x: 模拟非重叠最大池化，所有位置都是"最大值位置"
    # 这样 golden: dx = (x==y) ? dy : 0 = dy（所有元素通过）
    tmp_y[:original_size] = tmp_x[:original_size]

    # padding 区域全部为 0, x==y==0, dy==0 -> dx==0

    # golden 计算
    golden = max_pooling_grad_golden(tmp_dy, tmp_x, tmp_y)

    # 保存文件
    op_name = "max_pooling_grad"
    tmp_dy.astype(np_type).tofile(f"{d_type}_dy_t_{op_name}.bin")
    tmp_x.astype(np_type).tofile(f"{d_type}_x_t_{op_name}.bin")
    tmp_y.astype(np_type).tofile(f"{d_type}_y_t_{op_name}.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_dx_t_{op_name}.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
