#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

IGNORE_INDEX = -100


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    row_num = int(shape[0])
    class_num = int(shape[1])

    x = np.random.uniform(-2.0, 2.0, (row_num, class_num)).astype(np_type)
    target = np.random.randint(0, class_num, (row_num,)).astype(np.int32)

    # NLLLoss golden（无 weight，reduction=mean）：与算子内核保持一致的计算路径
    # loss_i = -x[i, target[i]]（ignore_index 的行不计入），mean = sum(loss_i) / valid_count
    picked = x[np.arange(row_num), target].astype(np.float32)
    valid = (target != IGNORE_INDEX).astype(np.float32)
    loss = -picked * valid
    total_weight = valid.sum()
    y = loss.sum() / total_weight if total_weight > 0 else np.float32(0.0)
    golden = np.array([y], dtype=np_type)

    x.astype(np_type).tofile(f"{d_type}_input_x_nll_loss.bin")
    target.astype(np.int32).tofile("int32_input_target_nll_loss.bin")
    golden.astype(np_type).tofile(f"{d_type}_golden_y_nll_loss.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
