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
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def layer_normalization_grad_golden(dy, x, gamma, mean, rstd):
    """实现 layer_normalization_grad golden 计算。

    公式（per-row）：
      x_hat = (x - mean) * rstd
      dxhat = dy * gamma
      ds    = sum(dxhat * x_hat)          # per-row scalar
      db    = sum(dxhat)                   # per-row scalar
      dx    = rstd * (dxhat - (db + ds * x_hat) / D)
      dgamma = sum(dy * x_hat, axis=0)    # cross-row
      dbeta  = sum(dy, axis=0)            # cross-row

    Args:
        dy:    [N, D] float32  — N = batch * pre_dims, D = normalized_dims
        x:     [N, D] float32
        gamma: [D] float32
        mean:  [N] float32
        rstd:  [N] float32

    Returns:
        dx:     [N, D] float32
        dgamma: [D] float32
        dbeta:  [D] float32
    """
    N, D = dy.shape

    # Flatten gamma if multi-dim
    gamma_flat = gamma.reshape(D)

    x_hat = (x - mean.reshape(N, 1)) * rstd.reshape(N, 1)  # [N, D]
    dxhat = dy * gamma_flat.reshape(1, D)  # [N, D]

    ds = np.sum(dxhat * x_hat, axis=1)  # [N]
    db = np.sum(dxhat, axis=1)  # [N]

    dx = rstd.reshape(N, 1) * (
        dxhat - (db.reshape(N, 1) + ds.reshape(N, 1) * x_hat) / float(D)
    )

    dgamma = np.sum(dy * x_hat, axis=0)  # [D]
    dbeta = np.sum(dy, axis=0)  # [D]

    return dx, dgamma, dbeta


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
    # [256, 128] → N = 256, D = 128
    N = int(shape[0])
    D = int(shape[1])
    total_spatial = N * D

    # 大张量值池（dy、x）
    val_pool = [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    dy = np.random.choice(val_pool, size=total_spatial).reshape(N, D)
    x = np.random.choice(val_pool, size=total_spatial).reshape(N, D)

    # per-feature gamma（D 维）
    param_pool = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    gamma = np.random.choice(param_pool, size=D)

    # per-row mean（可正可负）
    mean = np.random.choice(param_pool, size=N)

    # per-row rstd（恒正）
    invstd_pool = [0.25, 0.5, 1.0, 2.0]
    rstd = np.random.choice(invstd_pool, size=N)

    # golden 在 float32 下计算
    dy_fp32 = dy.astype(np.float32)
    x_fp32 = x.astype(np.float32)
    gamma_fp32 = gamma.astype(np.float32)
    mean_fp32 = mean.astype(np.float32)
    rstd_fp32 = rstd.astype(np.float32)

    golden_dx, golden_dgamma, golden_dbeta = layer_normalization_grad_golden(
        dy_fp32, x_fp32, gamma_fp32, mean_fp32, rstd_fp32
    )

    if d_type == "bfloat16":
        pack_as_bf16(dy).tofile("bfloat16_dy_t_layer_normalization_grad.bin")
        pack_as_bf16(x).tofile("bfloat16_x_t_layer_normalization_grad.bin")
        pack_as_bf16(gamma).tofile("bfloat16_gamma_t_layer_normalization_grad.bin")
        pack_as_bf16(mean).tofile("bfloat16_mean_t_layer_normalization_grad.bin")
        pack_as_bf16(rstd).tofile("bfloat16_rstd_t_layer_normalization_grad.bin")

        pack_as_bf16(golden_dx).tofile(
            "bfloat16_golden_dx_t_layer_normalization_grad.bin"
        )
        pack_as_bf16(golden_dgamma).tofile(
            "bfloat16_golden_dgamma_t_layer_normalization_grad.bin"
        )
        pack_as_bf16(golden_dbeta).tofile(
            "bfloat16_golden_dbeta_t_layer_normalization_grad.bin"
        )
    else:
        dy.astype(np_type).tofile(f"{d_type}_dy_t_layer_normalization_grad.bin")
        x.astype(np_type).tofile(f"{d_type}_x_t_layer_normalization_grad.bin")
        gamma.astype(np_type).tofile(f"{d_type}_gamma_t_layer_normalization_grad.bin")
        mean.astype(np_type).tofile(f"{d_type}_mean_t_layer_normalization_grad.bin")
        rstd.astype(np_type).tofile(f"{d_type}_rstd_t_layer_normalization_grad.bin")

        golden_dx.astype(np_type).tofile(
            f"{d_type}_golden_dx_t_layer_normalization_grad.bin"
        )
        golden_dgamma.astype(np_type).tofile(
            f"{d_type}_golden_dgamma_t_layer_normalization_grad.bin"
        )
        golden_dbeta.astype(np_type).tofile(
            f"{d_type}_golden_dbeta_t_layer_normalization_grad.bin"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type>")
        exit(1)
    # 清理旧 bin 文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
