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


def average_pooling_grad_golden(dy, kernel_size, strides, pads):
    """纯 numpy 实现 average_pooling_grad golden 计算：
    dx[n,c,h,w] = dy[n,c,h_out,w_out] / (kH*kW)
    其中 (h,w) 属于 (h_out,w_out) 对应的池化窗口 [h_out*sH - pH, h_out*sH - pH + kH)
    模拟 kernel 逐行写入行为（非累加，后续写入覆盖重叠区域）
    """
    N, C, H_out, W_out = dy.shape
    kH, kW = kernel_size
    sH, sW = strides
    pH, pW = pads
    H = (H_out - 1) * sH + kH - 2 * pH
    W = (W_out - 1) * sW + kW - 2 * pW

    dx = np.zeros((N, C, H, W), dtype=np.float32)
    inv_area = 1.0 / (kH * kW)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                h_start = h_out * sH - pH
                for w_out in range(W_out):
                    val = float(dy[n, c, h_out, w_out]) * inv_area
                    w_start = w_out * sW - pW
                    for i in range(kH):
                        h = h_start + i
                        if h < 0 or h >= H:
                            continue
                        for j in range(kW):
                            w = w_start + j
                            if w < 0 or w >= W:
                                continue
                            dx[n, c, h, w] = val
    return dx


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
    tmp_dy = np.random.choice(val_pool, size=size).reshape(shape)

    # kernel_size=[1,1], strides=[1,1], pads=[0,0]
    kernel_size = (1, 1)
    strides = (1, 1)
    pads = (0, 0)

    if d_type == "bfloat16":
        # golden 在 float32 下计算
        golden = average_pooling_grad_golden(
            tmp_dy.astype(np.float32), kernel_size, strides, pads
        )
        pack_as_bf16(tmp_dy).tofile(
            "bfloat16_dy_t_average_pooling_grad.bin"
        )
        pack_as_bf16(golden).tofile(
            "bfloat16_golden_dx_t_average_pooling_grad.bin"
        )
    else:
        tmp_dy = tmp_dy.astype(np_type)
        golden = average_pooling_grad_golden(
            tmp_dy.astype(np.float32), kernel_size, strides, pads
        )

        tmp_dy.tofile(f"{d_type}_dy_t_average_pooling_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_dx_t_average_pooling_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
