#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import os


def gen_golden_data():
    M = 1
    N = 12288
    K = 8192
    GROUP_SIZE = 128
    np.random.seed(1)

    input_a_fp16 = np.random.uniform(low=-3, high=3, size=(M, K)).astype(np.float16)
    input_antiquant_scale_fp16 = np.random.uniform(low=-3, high=3, size=(K // GROUP_SIZE, N)).astype(np.float16)
    golden = np.zeros((M, N)).astype(np.float32)

    input_weight_int4 = np.random.uniform(low=-7, high=7, size=(K, N)).astype(np.int8)
    # pack weight
    input_weight_int8_pack = np.zeros((K, N // 2)).astype(np.uint8)

    input_weight_int8_pack = (input_weight_int4.astype(np.uint8) << 4)[:, 1:N:2] + (
        input_weight_int4.astype(np.uint8) & 0x0F
    )[:, 0:N:2]

    f1 = np.float32(7.49)
    f2 = np.float32(14.98)
    const = np.float32(1)
    # unfold A
    for groupId in range(K // GROUP_SIZE):
        w_g = input_weight_int4.astype(np.int32)[groupId * GROUP_SIZE : (groupId + 1) * GROUP_SIZE, :].astype(np.int32)
        a_g = input_a_fp16[:, groupId * GROUP_SIZE : (groupId + 1) * GROUP_SIZE].astype(np.float32)
        a_max = np.max(np.abs(a_g), axis=1)

        a_tmp1 = a_g / a_max * f1
        a1 = np.round(a_tmp1)
        c1 = np.matmul(a1.astype(np.int32), w_g).astype(np.float32)
        a_tmp2 = (a_tmp1 - a1) * f2
        a2 = np.round(a_tmp2)
        c2 = np.matmul(a2.astype(np.int32), w_g).astype(np.float32)
        a_tmp3 = (a_tmp2 - a2) * f2
        a3 = np.round(a_tmp3)
        c3 = np.matmul(a3.astype(np.int32), w_g).astype(np.float32)
        c_tmp = c1 * (const / f1) + c2 * (const / (f1 * f2)) + c3 * (const / (f1 * f2 * f2))
        golden = golden + c_tmp * input_antiquant_scale_fp16[groupId : groupId + 1, :].astype(np.float32) * a_max

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")
    input_a_fp16.tofile("./input/input_a.bin")
    input_weight_int4.tofile("./input_input_weight_real.bin")
    input_weight_int8_pack.tofile("./input/input_weight.bin")
    input_antiquant_scale_fp16.tofile("./input/input_antiquant_scale_fp16.bin")
    golden.astype(np.float16).tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
