# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3

import sys
import numpy as np
import torch
import ast

def get_cpu_data(data_shape, input_predict, input_target, reduction):
    ori_type = input_predict.dtype
    input_predict = torch.from_numpy(input_predict)
    input_target = torch.from_numpy(input_target)
    output = torch.nn.functional.mse_loss(input_predict, input_target, reduction=reduction).numpy().astype(ori_type)
    if reduction == 'none':
        return output
    golden = np.zeros_like(data_shape)
    golden.ravel()[0] = output
    return golden

def gen_data_and_golden(data_shape, data_range, dtype, reduction):
    range_lo, range_hi = ast.literal_eval(data_range)
    data_shape = ast.literal_eval(data_shape)
    input_predict = np.random.uniform(range_lo, range_hi, data_shape).astype(dtype)
    input_target = np.random.uniform(range_lo, range_hi, data_shape).astype(dtype)
    output = get_cpu_data(data_shape, input_predict, input_target, reduction)
    # print(os.path.dirname(__file__))

    input_predict.tofile("./input_input.bin")
    input_target.tofile("./input_target.bin")

    output.tofile("./output_golden.bin")


if __name__ == "__main__":
    # print(f"gen data...")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])