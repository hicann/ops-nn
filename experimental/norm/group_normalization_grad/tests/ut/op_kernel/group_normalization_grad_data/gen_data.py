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


def group_normalization_grad_golden(x, dy, gamma, mean, rstd):
    """Pure numpy implementation of group_normalization_grad.

    Formula: dx = (rstd / M) * gamma * (M * dy - s1 - x_hat * s2)
    where:
      x_hat = (x - mean) * rstd
      s1 = sum(dy * gamma)  over each group
      s2 = sum(dy * gamma * x_hat)  over each group
      M = number of elements per group
    """
    # x, dy, gamma shape: [N, G, M] (N*G groups, M elements each)
    # mean, rstd shape: [N*G] (one scalar per group)
    N_G = x.shape[0] * x.shape[1]
    M = x.shape[2]
    dx = np.zeros_like(x, dtype=np.float32)

    x_f = x.astype(np.float32)
    dy_f = dy.astype(np.float32)
    gamma_f = gamma.astype(np.float32)
    mean_f = mean.astype(np.float32)
    rstd_f = rstd.astype(np.float32)

    for g in range(N_G):
        n = g // x.shape[1]
        c = g % x.shape[1]
        x_hat = (x_f[n, c, :] - mean_f[g]) * rstd_f[g]
        s1 = np.sum(dy_f[n, c, :] * gamma_f[n, c, :])
        s2 = np.sum(dy_f[n, c, :] * gamma_f[n, c, :] * x_hat)
        dx[n, c, :] = (
            (rstd_f[g] / M) * gamma_f[n, c, :] * (M * dy_f[n, c, :] - s1 - x_hat * s2)
        )

    return dx


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
    # shape format: [N, G, M] where N*G groups, M elements per group
    N_G = shape[0] * shape[1]

    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    tmp_x = np.random.choice(val_pool, size=np.prod(shape)).reshape(shape)
    tmp_dy = np.random.choice(val_pool, size=np.prod(shape)).reshape(shape)
    tmp_gamma = np.random.choice(val_pool, size=np.prod(shape)).reshape(shape)
    tmp_mean = np.random.choice(val_pool, size=N_G).reshape(N_G)
    tmp_rstd = np.abs(np.random.choice(val_pool, size=N_G).reshape(N_G)) + 0.1

    golden = group_normalization_grad_golden(
        tmp_x, tmp_dy, tmp_gamma, tmp_mean, tmp_rstd
    )

    if d_type == "bfloat16":
        pack_as_bf16(tmp_x).tofile("bfloat16_x_t_group_normalization_grad.bin")
        pack_as_bf16(tmp_dy).tofile("bfloat16_dy_t_group_normalization_grad.bin")
        pack_as_bf16(tmp_gamma).tofile("bfloat16_gamma_t_group_normalization_grad.bin")
        pack_as_bf16(tmp_mean).tofile("bfloat16_mean_t_group_normalization_grad.bin")
        pack_as_bf16(tmp_rstd).tofile("bfloat16_rstd_t_group_normalization_grad.bin")
        pack_as_bf16(golden).tofile("bfloat16_golden_t_group_normalization_grad.bin")
    else:
        tmp_x = tmp_x.astype(np_type)
        tmp_dy = tmp_dy.astype(np_type)
        tmp_gamma = tmp_gamma.astype(np_type)
        tmp_mean = tmp_mean.astype(np_type)
        tmp_rstd = tmp_rstd.astype(np_type)

        tmp_x.tofile(f"{d_type}_x_t_group_normalization_grad.bin")
        tmp_dy.tofile(f"{d_type}_dy_t_group_normalization_grad.bin")
        tmp_gamma.tofile(f"{d_type}_gamma_t_group_normalization_grad.bin")
        tmp_mean.tofile(f"{d_type}_mean_t_group_normalization_grad.bin")
        tmp_rstd.tofile(f"{d_type}_rstd_t_group_normalization_grad.bin")
        golden.astype(np_type).tofile(f"{d_type}_golden_t_group_normalization_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
