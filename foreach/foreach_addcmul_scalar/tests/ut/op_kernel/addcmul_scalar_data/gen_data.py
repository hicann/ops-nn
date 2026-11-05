#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

TAIL_MARKER = -999.0


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 ,]+)\}", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",")]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(
    shape_str, scale_value=2.0, d_type="float32", out_shape_str=None
):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    out_shape_list = (
        parse_str_to_shape_list(out_shape_str) if out_shape_str is not None else None
    )
    if d_type == "bfloat16_t":
        scalars = np.array([scale_value for i in range(len(shape_list))]).astype(
            np.float32
        )
    else:
        scalars = np.array([scale_value for i in range(len(shape_list))]).astype(
            np_type
        )
    for index, shape in enumerate(shape_list):
        tmp_input = np.ones(shape) * 10
        tmp_input = tmp_input.astype(np_type)
        tmp_tensor1 = np.ones(shape) * 10
        tmp_tensor1 = tmp_tensor1.astype(np_type)
        tmp_tensor2 = np.ones(shape) * 10
        tmp_tensor2 = tmp_tensor2.astype(np_type)
        tmp_golden = np.ones(shape) * 10
        tmp_golden = tmp_golden.astype(np_type)

        for i in range(len(tmp_golden)):
            tmp_golden[i] = (
                tmp_input[i] + tmp_tensor1[i] * tmp_tensor2[i] * scalars[index]
            )

        if out_shape_list is not None:
            # out 元素数大于 x1 的场景：golden 覆盖 out 全长（前 x1 numel 为计算值，尾部为标记值），
            # 另生成 outinit 文件（全长标记值）作为 out 的初始内存——kernel 只应写前 x1 numel 个元素
            out_numel = int(np.prod(out_shape_list[index]))
            x_numel = int(np.prod(shape))
            tmp_out_init = np.full((out_numel,), TAIL_MARKER, dtype=np_type)
            tmp_golden = np.full((out_numel,), TAIL_MARKER, dtype=np_type)
            # 用 flat 索引逐元素取值，兼容输入为多维 shape（如 {4,6}）的场景
            tmp_input_flat = tmp_input.flatten()
            tmp_tensor1_flat = tmp_tensor1.flatten()
            tmp_tensor2_flat = tmp_tensor2.flatten()
            for i in range(x_numel):
                tmp_golden[i] = (
                    tmp_input_flat[i]
                    + tmp_tensor1_flat[i] * tmp_tensor2_flat[i] * scalars[index]
                )

        tmp_golden = tmp_golden.astype(np_type)

        tmp_input.astype(np_type).tofile(f"{d_type}_input_t_foreach_addcmul{index}.bin")
        tmp_tensor1.astype(np_type).tofile(
            f"{d_type}_tensor1_t_foreach_addcmul{index}.bin"
        )
        tmp_tensor2.astype(np_type).tofile(
            f"{d_type}_tensor2_t_foreach_addcmul{index}.bin"
        )
        tmp_golden.astype(np_type).tofile(
            f"{d_type}_golden_t_foreach_addcmul{index}.bin"
        )
        if out_shape_list is not None:
            tmp_out_init.astype(np_type).tofile(
                f"{d_type}_outinit_t_foreach_addcmul{index}.bin"
            )


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Param num must be 4 or 5 (with optional out_shape).")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    out_shape_str = sys.argv[4] if len(sys.argv) == 5 else None
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], out_shape_str)
