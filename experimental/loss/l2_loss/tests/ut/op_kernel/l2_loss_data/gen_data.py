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
import tensorflow as tf

# 与算子内核保持完全一致的计算路径：先乘 1/√2 缩放，再平方，再求和
# 这是规定做法，golden 必须与算子路径对齐，否则因不同计算顺序导致的 ULP 差异会被当成算子错误
INV_SQRT2 = np.float32(0.70710678118654752)


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    size = np.prod(shape)
    tmp_input = np.random.choice([0, 0.5, 1, 2, 3, 10, 100], size=size)
    tmp_input = tmp_input.reshape(shape).astype(np_type)
    # L2Loss golden：与算子内核完全一致的路径 —— (x * INV_SQRT2)^2 逐元素计算后求和
    # 用 float32 精度模拟算子行为，避免 golden 与算子因精度路径不同产生系统性偏差
    scaled = tmp_input.astype(np.float32) * INV_SQRT2
    tmp_golden = np.sum(scaled * scaled)
    tmp_golden = np.array([tmp_golden], dtype=np_type)

    tmp_input.astype(np_type).tofile(f"{d_type}_input_t_l2_loss.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_t_l2_loss.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
