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


def hinge_loss_grad_golden(predict, target, grad_output):
    """纯 numpy 实现 hinge_loss_grad golden 计算：
    margin = 1 - target * predict
    grad_input = (margin > 0) ? (-target * grad_output) : 0
    """
    margin = 1.0 - target * predict
    return np.where(margin > 0.0, -target * grad_output, 0.0)


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
    val_pool = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    tmp_predict = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_target = np.random.choice(val_pool, size=size).reshape(shape)
    tmp_grad_output = np.random.choice(val_pool, size=size).reshape(shape)

    if d_type == "bfloat16":
        # golden 在 float32 下计算
        golden = hinge_loss_grad_golden(
            tmp_predict.astype(np.float32),
            tmp_target.astype(np.float32),
            tmp_grad_output.astype(np.float32)
        )
        pack_as_bf16(tmp_predict).tofile(
            "bfloat16_predict_t_hinge_loss_grad.bin"
        )
        pack_as_bf16(tmp_target).tofile(
            "bfloat16_target_t_hinge_loss_grad.bin"
        )
        pack_as_bf16(tmp_grad_output).tofile(
            "bfloat16_grad_output_t_hinge_loss_grad.bin"
        )
        pack_as_bf16(golden).tofile(
            "bfloat16_golden_grad_input_t_hinge_loss_grad.bin"
        )
    else:
        tmp_predict = tmp_predict.astype(np_type)
        tmp_target = tmp_target.astype(np_type)
        tmp_grad_output = tmp_grad_output.astype(np_type)
        golden = hinge_loss_grad_golden(
            tmp_predict.astype(np.float32),
            tmp_target.astype(np.float32),
            tmp_grad_output.astype(np.float32)
        )

        tmp_predict.tofile(f"{d_type}_predict_t_hinge_loss_grad.bin")
        tmp_target.tofile(f"{d_type}_target_t_hinge_loss_grad.bin")
        tmp_grad_output.tofile(f"{d_type}_grad_output_t_hinge_loss_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_grad_input_t_hinge_loss_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
