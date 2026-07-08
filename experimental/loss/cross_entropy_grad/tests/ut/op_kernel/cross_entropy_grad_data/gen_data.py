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


def cross_entropy_grad_golden(logits, target):
    """纯 numpy 实现 cross_entropy_grad golden 计算：
    grad = softmax(logits) - one_hot(target)
    target[i] 是第 i 行的正确类别索引
    """
    logits_f32 = logits.astype(np.float32)
    num_classes = logits_f32.shape[-1]

    # stable softmax
    max_vals = np.max(logits_f32, axis=-1, keepdims=True)
    exp_vals = np.exp(logits_f32 - max_vals)
    softmax_vals = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    # one_hot
    one_hot = np.zeros_like(softmax_vals)
    rows = logits_f32.shape[0]
    for i in range(rows):
        one_hot[i, int(target[i])] = 1.0

    return softmax_vals - one_hot


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
    num_rows = shape[0]
    num_classes = shape[1]

    # 生成随机测试数据
    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0]
    tmp_logits = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_target = np.random.randint(0, num_classes, size=num_rows)

    if d_type == "bfloat16":
        # golden 在 float32 下计算
        golden = cross_entropy_grad_golden(tmp_logits, tmp_target)
        pack_as_bf16(tmp_logits).tofile(
            "bfloat16_logits_t_cross_entropy_grad.bin"
        )
        tmp_target.astype(np.int32).tofile(
            "bfloat16_target_t_cross_entropy_grad.bin"
        )
        pack_as_bf16(golden).tofile(
            "bfloat16_golden_t_cross_entropy_grad.bin"
        )
    else:
        tmp_logits = tmp_logits.astype(np_type)
        golden = cross_entropy_grad_golden(tmp_logits, tmp_target)

        tmp_logits.tofile(f"{d_type}_logits_t_cross_entropy_grad.bin")
        tmp_target.astype(np.int32).tofile(f"{d_type}_target_t_cross_entropy_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_t_cross_entropy_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
