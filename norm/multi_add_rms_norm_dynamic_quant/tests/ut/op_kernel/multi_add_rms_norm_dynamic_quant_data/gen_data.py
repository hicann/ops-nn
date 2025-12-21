#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------

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


def gen_data_and_golden(nd_shape_str, length, d_type="float16"):
    d_type_dict = {
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    nd_shape_list = parse_str_to_shape_list(nd_shape_str)
    
    for i in range(0, int(length)):
        x1 = np.array(nd_shape_list).astype(np_type)
        x1.tofile(f"{d_type}_x1_{i}.bin")
    x2 = np.array(nd_shape_list).astype(np_type)
    x2.tofile(f"x2.bin")
    gamma = np.array(nd_shape_list[-1:]).astype(np_type)
    gamma.tofile(f"gamma.bin")
    smooth1 = np.array(nd_shape_list[-1:]).astype(np_type)
    smooth1.tofile(f"smooth1.bin")
    smooth2 = np.array(nd_shape_list[-1:]).astype(np_type)
    smooth2.tofile(f"smooth2.bin")

    # todogen golden


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])