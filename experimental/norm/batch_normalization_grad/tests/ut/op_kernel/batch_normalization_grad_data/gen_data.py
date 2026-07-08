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


def batch_normalization_grad_golden(grad_output, input_data, weight, bias,
                                     save_mean, save_invstd, epsilon=1e-5):
    """纯 numpy 实现 batch_normalization_grad golden 计算。

    公式（per-channel）：
      xhat = (input - save_mean) * save_invstd
      g_bar = mean(grad_output)
      gxhat_bar = mean(grad_output * xhat)
      grad_input = weight * save_invstd * (grad_output - g_bar - xhat * gxhat_bar)
      grad_weight = sum(grad_output * xhat)
      grad_bias = sum(grad_output)

    Args:
        grad_output: [N, C, H, W] float32
        input_data:  [N, C, H, W] float32
        weight:      [C] float32
        bias:        [C] float32 (unused in computation, kept for API consistency)
        save_mean:   [C] float32
        save_invstd: [C] float32
        epsilon:     float (unused in gradient formula, kept for API consistency)

    Returns:
        grad_input:  [N, C, H, W] float32
        grad_weight: [C] float32
        grad_bias:   [C] float32
    """
    N, C, H, W = grad_output.shape
    grad_input = np.zeros_like(grad_output, dtype=np.float32)
    grad_weight = np.zeros(C, dtype=np.float32)
    grad_bias = np.zeros(C, dtype=np.float32)

    for c in range(C):
        go_c = grad_output[:, c, :, :]        # [N, H, W]
        in_c = input_data[:, c, :, :]          # [N, H, W]

        xhat = (in_c - save_mean[c]) * save_invstd[c]
        g_bar = np.mean(go_c)
        gxhat_bar = np.mean(go_c * xhat)

        grad_input[:, c, :, :] = (
            weight[c] * save_invstd[c] * (go_c - g_bar - xhat * gxhat_bar)
        )
        grad_weight[c] = np.sum(go_c * xhat)
        grad_bias[c] = np.sum(go_c)

    return grad_input, grad_weight, grad_bias


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
    N, C, H, W = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    total_spatial = N * C * H * W

    # 大张量值池（grad_output、input）
    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    grad_out = np.random.choice(val_pool, size=total_spatial).reshape(N, C, H, W)
    input_data = np.random.choice(val_pool, size=total_spatial).reshape(N, C, H, W)

    # per-channel 参数值池
    param_pool = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    weight = np.random.choice(param_pool, size=C)
    bias = np.random.choice(param_pool, size=C)
    save_mean = np.random.choice(param_pool, size=C)

    # save_invstd 恒正
    invstd_pool = [0.25, 0.5, 1.0, 2.0]
    save_invstd = np.random.choice(invstd_pool, size=C)

    # golden 在 float32 下计算
    grad_out_fp32 = grad_out.astype(np.float32)
    input_fp32 = input_data.astype(np.float32)
    weight_fp32 = weight.astype(np.float32)
    bias_fp32 = bias.astype(np.float32)
    save_mean_fp32 = save_mean.astype(np.float32)
    save_invstd_fp32 = save_invstd.astype(np.float32)

    golden_gi, golden_gw, golden_gb = batch_normalization_grad_golden(
        grad_out_fp32, input_fp32, weight_fp32, bias_fp32,
        save_mean_fp32, save_invstd_fp32
    )

    if d_type == "bfloat16":
        pack_as_bf16(grad_out).tofile("bfloat16_grad_output_t_batch_normalization_grad.bin")
        pack_as_bf16(input_data).tofile("bfloat16_input_t_batch_normalization_grad.bin")
        pack_as_bf16(weight).tofile("bfloat16_weight_t_batch_normalization_grad.bin")
        pack_as_bf16(bias).tofile("bfloat16_bias_t_batch_normalization_grad.bin")
        pack_as_bf16(save_mean).tofile("bfloat16_save_mean_t_batch_normalization_grad.bin")
        pack_as_bf16(save_invstd).tofile("bfloat16_save_invstd_t_batch_normalization_grad.bin")

        pack_as_bf16(golden_gi).tofile("bfloat16_golden_grad_input_t_batch_normalization_grad.bin")
        pack_as_bf16(golden_gw).tofile("bfloat16_golden_grad_weight_t_batch_normalization_grad.bin")
        pack_as_bf16(golden_gb).tofile("bfloat16_golden_grad_bias_t_batch_normalization_grad.bin")
    else:
        grad_out.astype(np_type).tofile(f"{d_type}_grad_output_t_batch_normalization_grad.bin")
        input_data.astype(np_type).tofile(f"{d_type}_input_t_batch_normalization_grad.bin")
        weight.astype(np_type).tofile(f"{d_type}_weight_t_batch_normalization_grad.bin")
        bias.astype(np_type).tofile(f"{d_type}_bias_t_batch_normalization_grad.bin")
        save_mean.astype(np_type).tofile(f"{d_type}_save_mean_t_batch_normalization_grad.bin")
        save_invstd.astype(np_type).tofile(f"{d_type}_save_invstd_t_batch_normalization_grad.bin")

        golden_gi.astype(np_type).tofile(f"{d_type}_golden_grad_input_t_batch_normalization_grad.bin")
        golden_gw.astype(np_type).tofile(f"{d_type}_golden_grad_weight_t_batch_normalization_grad.bin")
        golden_gb.astype(np_type).tofile(f"{d_type}_golden_grad_bias_t_batch_normalization_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
