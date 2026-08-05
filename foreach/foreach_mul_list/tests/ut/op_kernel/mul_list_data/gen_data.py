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
import re
import ml_dtypes


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16_t": ml_dtypes.bfloat16,
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        rng = np.random.default_rng(20260715 + index)
        if d_type == "int16":
            tmp_input_1 = rng.integers(-32768, 32768, size=shape, dtype=np.int16)
            tmp_input_2 = rng.integers(-32768, 32768, size=shape, dtype=np.int16)
        elif d_type == "int8":
            tmp_input_1 = rng.integers(-128, 128, size=shape, dtype=np.int8)
            tmp_input_2 = rng.integers(-128, 128, size=shape, dtype=np.int8)
        elif d_type == "uint8":
            tmp_input_1 = rng.integers(0, 256, size=shape, dtype=np.uint8)
            tmp_input_2 = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif d_type == "int32":
            tmp_input_1 = rng.integers(-1000, 1001, size=shape, dtype=np.int32)
            tmp_input_2 = rng.integers(-1000, 1001, size=shape, dtype=np.int32)
        else:
            tmp_input_1 = rng.uniform(-10.0, 10.0, size=shape).astype(np_type)
            tmp_input_2 = rng.uniform(-10.0, 10.0, size=shape).astype(np_type)

        tmp_golden = tmp_input_1.astype(np_type) * tmp_input_2.astype(np_type)

        tmp_input_1.astype(np_type).tofile(f"{d_type}_input_t_foreach_mul{index}_1.bin")
        tmp_input_2.astype(np_type).tofile(f"{d_type}_input_t_foreach_mul{index}_2.bin")
        tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_foreach_mul{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
