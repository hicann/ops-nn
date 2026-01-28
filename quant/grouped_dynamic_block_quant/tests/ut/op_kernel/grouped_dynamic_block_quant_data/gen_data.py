#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf

bfp16 = tf.bfloat16.as_numpy_dtype


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list


def gen_data(x_shape_str, group_num_str, d_type="float16"):
    d_type_dict = {
        "float16": np.float16,
        "bfloat16": bfp16
    }
    torch_type_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    np_type = d_type_dict[d_type]
    torch_type = torch_type_dict[d_type]
    
    x_shape, _ = parse_str_to_shape_list(x_shape_str)
    group_num = int(group_num_str)

    group_list = np.random.choice(np.arange(1, x_shape[0] + 1), size=group_num, replace=False)

    tem_x = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np_type)
    
    x_tensor = torch.tensor(tem_x, dtype=torch_type)

    tem_x.astype(np_type).tofile(f"input_size.bin")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data(sys.argv[1], sys.argv[2], sys.argv[3])