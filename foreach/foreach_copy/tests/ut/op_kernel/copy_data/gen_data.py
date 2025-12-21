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
import tensorflow as tf


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
        "bfloat16_t": tf.bfloat16.as_numpy_dtype,
        "int8_t": np.int8,
        "uint8_t": np.uint8,
        "int16_t": np.int16,
        "uint16_t": np.uint16,
        "int32_t": np.int32,
        "uint32_t": np.uint32,
        "int64_t" : np.int64,
        "float64" : np.float64,
        "bool" : np.bool_
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        tmp_input = np.random.rand(*shape) * 2 - 1
        tmp_input = tmp_input.astype(np_type)
        tmp_golden = np.random.rand(*shape) * 2 - 1
        tmp_golden = tmp_golden.astype(np_type)
        
        tmp_golden = tmp_input.copy()

        tmp_input.astype(np_type).tofile(f"{d_type}_input_t_foreach_copy_{index}.bin")
        tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_foreach_copy_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
