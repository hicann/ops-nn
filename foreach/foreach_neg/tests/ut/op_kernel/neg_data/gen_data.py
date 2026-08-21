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
import numpy as np
import re


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
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    np_type = d_type_dict.get(d_type)
    edge_value_dict = {
        "int16": [-32768, -32767, -1, 0, 1, 32767],
        "int8": [-128, -127, -1, 0, 1, 127],
        "uint8": [0, 1, 2, 127, 128, 255],
    }
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        if d_type == "bfloat16_t":
            tmp_input = (np.random.rand(*shape) * 2 - 1).astype(np.float32)
            input_bits = tmp_input.view(np.uint32)
            rounding_bias = np.uint32(0x7FFF) + ((input_bits >> 16) & 1)
            input_data = ((input_bits + rounding_bias) >> 16).astype(np.uint16)
            golden_data = input_data ^ np.uint16(0x8000)
        elif d_type in ("int32", "int16", "int8"):
            tmp_input = np.random.randint(-100, 100, size=shape).astype(np_type)
        elif d_type == "uint8":
            tmp_input = np.random.randint(0, 100, size=shape).astype(np_type)
        else:
            tmp_input = np.random.rand(*shape) * 2 - 1

        if d_type != "bfloat16_t":
            if d_type in edge_value_dict:
                flat_input = tmp_input.reshape(-1)
                edge_values = np.array(edge_value_dict[d_type], dtype=np_type)
                edge_count = min(flat_input.size, edge_values.size)
                flat_input[:edge_count] = edge_values[:edge_count]

            input_data = tmp_input.astype(np_type)
            golden_data = (-tmp_input).astype(np_type)

        input_data.tofile(f"{d_type}_input_t_foreach_neg_{index}.bin")
        golden_data.tofile(f"{d_type}_golden_t_foreach_neg_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
