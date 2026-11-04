#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import sys
import os
import numpy as np
import re
import tensorflow as tf
import torch


def _fp32_to_bf16_bits(arr):
    """float32 -> bfloat16 载体(uint16), 采用 round-to-nearest-even, 与内核 Cast 口径一致。"""
    bits = arr.astype(np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def _bf16_bits_to_fp32(arr):
    """bfloat16 载体(uint16) -> float32, 按位左移还原, 不引入额外精度损失。"""
    return (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def golden_sigmoid(tmp_input, np_type):
    """参考结果由 torch 计算(竞品优先), 不经 NumPy 做参考数学。
    与内核 ComputeSigmoid 一致: fp16/bf16 先升 fp32 计算, 结果再窄回输出 dtype。"""
    if np_type == np.float16:
        t = torch.from_numpy(np.asarray(tmp_input, dtype=np.float16))
        return torch.sigmoid(t.float()).half().numpy()
    if np_type == np.int32:
        t = torch.from_numpy(np.asarray(tmp_input, dtype=np.int32))
        return torch.sigmoid(t.float()).to(torch.int32).numpy()
    if np_type not in (np.float32,):
        # bfloat16_t: torch 不支持 bf16 直接落 numpy, 走 fp32 计算后按位窄回 uint16 载体
        src = _bf16_bits_to_fp32(np.asarray(tmp_input))
        res = torch.sigmoid(torch.from_numpy(src)).numpy()
        return _fp32_to_bf16_bits(res)
    return torch.sigmoid(
        torch.from_numpy(np.asarray(tmp_input, dtype=np.float32))
    ).numpy()


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
        "int32": np.int32,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype,
    }
    np_type = d_type_dict[d_type]
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        tmp_input = np.random.rand(*shape)
        tmp_input = tmp_input.astype(np_type)
        tmp_golden = golden_sigmoid(tmp_input, np_type).astype(np_type)

        tmp_input.astype(np_type).tofile(f"{d_type}_input_t_foreach_sigmoid{index}.bin")
        tmp_golden.astype(np_type).tofile(
            f"{d_type}_golden_t_foreach_sigmoid{index}.bin"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
