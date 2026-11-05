#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import random
import re
import struct


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(shape_str, d_type="float32"):
    type_info = {
        "float32": ("float32", 4),
        "float16": ("float16", 2),
        "int16": ("int16", 2),
        "int32": ("int32", 4),
        "bfloat16_t": ("bfloat16", 2),
    }
    if d_type not in type_info:
        raise ValueError(f"Unsupported dtype: {d_type}")

    shape_list = parse_str_to_shape_list(shape_str)
    rng = random.Random(0)
    for index, shape in enumerate(shape_list):
        element_count = 1
        for dimension in shape:
            element_count *= dimension

        values = [rng.uniform(-1.0, 1.0) for _ in range(element_count)]
        input_data = b"".join(pack_value(d_type, value) for value in values)
        golden_data = b"".join(pack_value(d_type, 0) for _ in values)

        with open(
            f"{d_type}_input_t_foreach_zero_inplace_{index}.bin", "wb"
        ) as input_file:
            input_file.write(input_data)
        with open(
            f"{d_type}_golden_t_foreach_zero_inplace_{index}.bin", "wb"
        ) as golden_file:
            golden_file.write(golden_data)


def pack_value(d_type, value):
    if d_type == "float32":
        return struct.pack("<f", value)
    if d_type == "float16":
        return struct.pack("<e", value)
    if d_type == "int32":
        return struct.pack("<i", int(value * 1000))
    if d_type == "int16":
        return struct.pack("<h", int(value * 100))
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.pack("<H", bits >> 16)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
