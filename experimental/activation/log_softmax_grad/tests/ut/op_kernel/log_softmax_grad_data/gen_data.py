#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import numpy as np
import torch

# Fixed seed so the generated test data is deterministic across runs (avoids flaky
# pass/fail caused by different random inputs hitting near-zero cancellation).
np.random.seed(1234)


def gen_data(shape, axis, dtype):
    # Generate data in the ORIGINAL operator shape (row-major, exactly as the GM
    # buffer the kernel sees). The kernel performs the reduction over the real
    # reduce axis in physical memory, so reducing over the same `axis` here makes
    # the golden match the kernel result for every tiling mode (NO_NEED_REDUCE /
    # REDUCE_TAIL / REDUCE_MID) regardless of how the tiling remaps merged dims.
    np_dtype = np.float32

    total = 1
    for s in shape:
        total *= s

    # grad_output (dy) and the forward logits (z) are random.
    data_dy = np.random.uniform(-3.0, 3.0, total).astype(np_dtype)
    data_z = np.random.uniform(-3.0, 3.0, total).astype(np_dtype)

    dy_r = torch.tensor(data_dy.reshape(shape))
    z_r = torch.tensor(data_z.reshape(shape))

    # Per the kernel contract, input x must be the (already normalized) log_softmax(z)
    # output so that exp(x) sums to 1 over the reduce axis.
    reduce_dim = axis[0] if len(axis) == 1 else tuple(axis)
    x_r = torch.log_softmax(z_r, dim=reduce_dim)

    # Kernel math (op_kernel/no_need_reduce.h etc.):
    #   z_out = dy - exp(x) * sum(dy)   (reduce over axis)
    # With exp(x) summing to 1, this equals the true log_softmax gradient.
    exp_x_r = torch.exp(x_r)
    sum_dy_r = dy_r.sum(dim=reduce_dim, keepdim=True)
    golden_r = dy_r - exp_x_r * sum_dy_r

    data_x = x_r.numpy().reshape(-1).astype(np_dtype)
    golden = golden_r.numpy().reshape(-1).astype(np_dtype)

    data_dy.tofile("./input_dy.bin")
    data_x.tofile("./input_x.bin")
    golden.tofile("./golden.bin")


if __name__ == "__main__":
    shape = [int(x) for x in sys.argv[1].split(",")]
    axis = [int(x) for x in sys.argv[2].split(",")]
    dtype = sys.argv[3]
    gen_data(shape, axis, dtype)
