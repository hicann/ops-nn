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
        "int16": np.int16,
        "int8": np.int8,
        "uint8": np.uint8,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        if d_type == "uint8":
            tmp_input = np.random.randint(0, 5, size=shape).astype(np.float32)
        elif d_type in ["int16", "int8"]:
            tmp_input = np.random.randint(-5, 5, size=shape).astype(np.float32)
        else:
            tmp_input = np.random.rand(*shape) * 2 - 1

        if d_type in ["int16", "int8", "uint8"]:
            tmp_golden = np.round(np.exp(tmp_input)).astype(np_type)
        else:
            tmp_golden = np.exp(tmp_input).astype(np_type)

        tmp_input.astype(np_type).tofile(f"{d_type}_input_t_foreach_exp_{index}.bin")
        tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_foreach_exp_{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
