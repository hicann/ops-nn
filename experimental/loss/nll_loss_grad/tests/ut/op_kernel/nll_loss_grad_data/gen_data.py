# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
    return shape_list


def nll_loss_grad_golden(
    target, weight, total_weight, y_grad, N, C, reduction, ignore_index
):
    x_grad = np.zeros((N, C), dtype=np.float32)
    for i in range(N):
        t = int(target[i])
        if t == ignore_index:
            continue
        if reduction == "none":
            scale = -float(y_grad[i])
        elif reduction == "sum":
            scale = -float(y_grad[0])
        else:
            if abs(float(total_weight[0])) < 1e-30:
                scale = 0.0
            else:
                scale = -float(y_grad[0]) / float(total_weight[0])
        x_grad[i, t] = scale * float(weight[t])
    return x_grad


def pack_as_bf16(arr_fp32):
    tmp = np.asarray(arr_fp32, dtype=np.float32)
    view_32 = tmp.view(np.uint32)
    bf16_u16 = (view_32 >> 16).astype(np.uint16)
    return bf16_u16


def gen_data_and_golden(
    shape_str, d_type="float32", reduction="mean", ignore_index=-100
):
    shape = parse_str_to_shape_list(shape_str)
    if len(shape) == 1:
        N, C = 1, shape[0]
    else:
        N, C = shape[0], shape[1]

    np.random.seed(42)

    x = np.random.randn(N, C).astype(np.float32)
    target = np.random.randint(0, C, size=(N,)).astype(np.int32)
    weight = np.abs(np.random.randn(C).astype(np.float32)) + 0.1

    valid_mask = target != ignore_index
    total_weight_val = np.sum(weight[target[valid_mask]])
    total_weight = np.array([total_weight_val], dtype=np.float32)

    if reduction == "none":
        y_grad = np.random.randn(N).astype(np.float32)
    else:
        y_grad = np.random.randn(1).astype(np.float32)

    golden = nll_loss_grad_golden(
        target, weight, total_weight, y_grad, N, C, reduction, ignore_index
    )

    if d_type == "bfloat16":
        pack_as_bf16(x).tofile("bfloat16_x_t_nll_loss_grad.bin")
        pack_as_bf16(y_grad).tofile("bfloat16_y_grad_t_nll_loss_grad.bin")
        target.tofile("bfloat16_target_t_nll_loss_grad.bin")
        pack_as_bf16(weight).tofile("bfloat16_weight_t_nll_loss_grad.bin")
        pack_as_bf16(total_weight).tofile("bfloat16_total_weight_t_nll_loss_grad.bin")
        pack_as_bf16(golden).tofile("bfloat16_golden_x_grad_t_nll_loss_grad.bin")
    else:
        np_type = np.float32 if d_type == "float32" else np.float16
        x.astype(np_type).tofile(f"{d_type}_x_t_nll_loss_grad.bin")
        y_grad.astype(np_type).tofile(f"{d_type}_y_grad_t_nll_loss_grad.bin")
        target.tofile(f"{d_type}_target_t_nll_loss_grad.bin")
        weight.astype(np_type).tofile(f"{d_type}_weight_t_nll_loss_grad.bin")
        total_weight.astype(np_type).tofile(
            f"{d_type}_total_weight_t_nll_loss_grad.bin"
        )
        golden.astype(np_type).tofile(f"{d_type}_golden_x_grad_t_nll_loss_grad.bin")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: gen_data.py <shape_str> <d_type> [reduction] [ignore_index]")
        exit(1)
    os.system("rm -rf *.bin")
    shape_str = sys.argv[1]
    d_type = sys.argv[2]
    reduction = sys.argv[3] if len(sys.argv) > 3 else "mean"
    ignore_index = int(sys.argv[4]) if len(sys.argv) > 4 else -100
    gen_data_and_golden(shape_str, d_type, reduction, ignore_index)
