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
import torch
import tensorflow as tf

bfp16 = tf.bfloat16.as_numpy_dtype

def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def do_golden(x, kronecker_p1, kronecker_p2):
    x1 = x @ kronecker_p2
    x2 = kronecker_p1 @ x1
    x2 = x2.flatten(-2,-1)
    quant_scale = x2.max(dim=-1, keepdim=True)[0].to(dtype=torch.float)/7
    return quant_scale

def gen_data_and_golden(x_shape_str, kronecker_p1_shape_str, kronecker_p2_shape_str, d_type="float16"):
    d_type_dict = {
        "float16": np.float16,
        "bfloat16": bfp16,
    }
    torch_type_dict = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    np_type = d_type_dict[d_type]
    torch_type = torch_type_dict[d_type]

    x_shape, _ = parse_str_to_shape_list(x_shape_str)
    kronecker_p1_shape, _ = parse_str_to_shape_list(kronecker_p1_shape_str)
    kronecker_p2_shape, _ = parse_str_to_shape_list(kronecker_p2_shape_str)

    tem_x = np.random.random(np.prod(x_shape)).reshape(x_shape).astype(np_type)
    tem_kronecker_p1 = np.random.random(np.prod(kronecker_p1_shape)).reshape(kronecker_p1_shape).astype(np_type)
    tem_kronecker_p2 = np.random.random(np.prod(kronecker_p2_shape)).reshape(kronecker_p2_shape).astype(np_type)

    x_tensor = torch.tensor(tem_x, dtype=torch_type)
    kronecker_p1_tensor = torch.tensor(tem_kronecker_p1, dtype=torch_type)
    kronecker_p2_tensor = torch.tensor(tem_kronecker_p2, dtype=torch_type)

    quant_scale = do_golden(x_tensor.to(torch.float), kronecker_p1_tensor.to(torch.float), kronecker_p2_tensor.to(torch.float))

    tmp_quant_scale_golden = np.array(quant_scale.cpu()).astype(np.float32)

    tem_x.astype(np_type).tofile(f"{d_type}_input_x_flat_quant.bin")
    tem_kronecker_p1.tofile(f"{d_type}_input_kronecker_p1_flat_quant.bin")
    tem_kronecker_p2.tofile(f"{d_type}_input_kronecker_p2_flat_quant.bin")

    tmp_quant_scale_golden.tofile(f"{d_type}_golden_quant_scale_flat_quant.bin")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Param num must be 5, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
