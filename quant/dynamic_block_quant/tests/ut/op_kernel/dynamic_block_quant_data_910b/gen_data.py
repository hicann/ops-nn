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


def do_golden(input):
    if len(input.shape) == 3:
        b0, m0, n0 = input.shape
        m = b0 * m0
        n = n0
    elif len(input.shape) == 2:
        m, n = input.shape
    pad_size = (128 - (n % 128)) % 128
    x = input.reshape((m, n))
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1)
    y_data = x_view * (127.0 / x_amax.unsqueeze(2))
    y_data = y_data.round()
    y_data = torch.clamp(y_data, -127, 127).to(torch.int8)
    y_data = y_data.view(m, n + pad_size)[:, :n].reshape(input.shape)
    scale = x_amax / 127.0
    if len(input.shape) == 3:
        scale = scale.reshape((b0, m0, -1))
    else:
        scale = scale.reshape((m, -1))
    return y_data, scale


def gen_data_and_golden(x_shape_str, d_type="float16"):
    d_type_dict = {
        "float16": np.float16,
        "bfloat16": bfp16,
        "int8": np.int8,
    }
    torch_type_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
    }
    np_type = d_type_dict[d_type]
    torch_type = torch_type_dict[d_type]
    
    x_shape, _ = parse_str_to_shape_list(x_shape_str)

    tem_x = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np_type)
    
    x_tensor = torch.tensor(tem_x, dtype=torch_type)
    
    y, scale = do_golden(x_tensor)
    
    tmp_y_golden = np.array(y.cpu()).astype(np.int8)
    tmp_scale_golden = np.array(scale.cpu()).astype(np.float32)

    tem_x.astype(np_type).tofile(f"{d_type}_input_x_dynamic_block_quant.bin")

    tmp_scale_golden.tofile(f"{d_type}_golden_scale_dynamic_block_quant.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
