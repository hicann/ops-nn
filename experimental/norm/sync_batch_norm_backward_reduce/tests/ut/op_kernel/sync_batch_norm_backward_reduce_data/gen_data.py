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
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")").rstrip(",")
    shape_list = [int(x) for x in shape_str.split(",") if x.strip() != ""]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = int(np.prod(shape))

    sum_dy = np.random.uniform(-1.0, 1.0, size).astype(np_type)
    sum_dy_dx_pad = np.random.uniform(-1.0, 1.0, size).astype(np_type)
    mean = np.random.uniform(-1.0, 1.0, size).astype(np_type)
    invert_std = np.random.uniform(0.5, 1.5, size).astype(np_type)

    # SyncBatchNormBackwardReduce golden（与算子内核保持一致的计算路径）：
    #   sum_dy_xmu = sum_dy_dx_pad - mean * sum_dy
    #   y (grad_weight) = sum_dy_xmu * invert_std
    sum_dy_f = sum_dy.astype(np.float32)
    sum_dy_dx_pad_f = sum_dy_dx_pad.astype(np.float32)
    mean_f = mean.astype(np.float32)
    invert_std_f = invert_std.astype(np.float32)
    sum_dy_xmu = sum_dy_dx_pad_f - mean_f * sum_dy_f
    y = sum_dy_xmu * invert_std_f

    sum_dy.astype(np_type).tofile(f"{d_type}_input_sum_dy.bin")
    sum_dy_dx_pad.astype(np_type).tofile(f"{d_type}_input_sum_dy_dx_pad.bin")
    mean.astype(np_type).tofile(f"{d_type}_input_mean.bin")
    invert_std.astype(np_type).tofile(f"{d_type}_input_invert_std.bin")
    sum_dy_xmu.astype(np_type).tofile(f"{d_type}_golden_sum_dy_xmu.bin")
    y.astype(np_type).tofile(f"{d_type}_golden_y.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
