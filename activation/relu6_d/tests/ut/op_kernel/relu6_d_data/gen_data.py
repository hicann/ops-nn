#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import numpy as np

SCALE = 1.0  # scale=1.0 → upper threshold = 6.0


def relu6_d_fp32(x):
    return np.minimum(np.maximum(x, 0), 6 * SCALE)


def relu6_d_int32(x):
    return np.minimum(np.maximum(x, 0), int(6 * SCALE))


def gen_golden_data_simple(shape_str, dtype):
    dtype_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16": None,
    }
    np_dtype = dtype_dict[dtype]

    shape_list = [int(x) for x in shape_str.strip("(").strip(")").split(",")]
    shape = tuple(shape_list)

    if dtype == "bfloat16":
        import ml_dtypes

        data_x = np.random.uniform(-10, 10, shape).astype(np.float32)
        golden = relu6_d_fp32(data_x)
        data_x_bf16 = data_x.astype(ml_dtypes.bfloat16)
        golden_bf16 = golden.astype(ml_dtypes.bfloat16)
        data_x_bf16.tofile("./input_x.bin")
        golden_bf16.tofile("./golden.bin")
    elif dtype == "int32":
        data_x = np.random.randint(-10, 10, shape).astype(np.int32)
        golden = relu6_d_int32(data_x)
        data_x.tofile("./input_x.bin")
        golden.tofile("./golden.bin")
    else:
        data_x = np.random.uniform(-10, 10, shape).astype(np_dtype)
        golden = relu6_d_fp32(data_x.astype(np.float32)).astype(np_dtype)
        data_x.tofile("./input_x.bin")
        golden.tofile("./golden.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gen_data.py <shape> <dtype>")
        print("Example: python gen_data.py '(64)' float32")
        exit(1)
    gen_golden_data_simple(sys.argv[1], sys.argv[2])
