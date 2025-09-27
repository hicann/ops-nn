#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import os
import numpy as np
import re
import tensorflow as tf
import torch
import stat
import traceback


OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 \,]+)\}\,", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",") if x.strip()]
        shape_list.append(single_shape)
    return shape_list

def gen_data(shape_str, attr_str, tiling_key="1"):
    if tiling_key in ["3", "103"]:
        d_type = tf.bfloat16.as_numpy_dtype
    elif tiling_key in ["2", "102"]:
        d_type = np.float16
    elif tiling_key in ["1", "101"]:
        d_type = np.float32
    elif tiling_key in ["4", "104"]:
        d_type = np.int16
    else:
        d_type = np.int32

    shape_list = parse_str_to_shape_list(shape_str)

    data_var = np.random.uniform(10, 20, tuple(shape_list[0])).astype(d_type)
    data_var.tofile(f"var.bin")
    data_value = np.random.uniform(10, 20, tuple(shape_list[1])).astype(d_type)
    data_value.tofile(f"value.bin")
    data_index = np.random.randint(50, tuple(shape_list[2])).astype(np.int32)

    tensor_index = torch.from_numpy(data_index)
    index, pos = torch.sort(tensor_index)
    index = index.to(torch.int32).numpy()
    pos = pos.to(torch.int32).numpy()
    index.tofile("./index.bin")
    pos.tofile("./pos.bin")
    data_alpha = np.random.uniform(0, 10, 1).astype(d_type)
    if d_type == np.float16 or d_type == tf.bfloat16.as_numpy_dtype:
        data_alpha.astype(np.float32)
    data_alpha.tofile("./alpha.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    try:
        gen_data(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception:
        traceback.print_exc()
        exit(2)
    exit(0)
