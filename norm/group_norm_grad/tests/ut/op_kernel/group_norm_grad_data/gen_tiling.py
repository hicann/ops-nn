#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/
import numpy as np
import sys


def GetDataTypeSize(dType_str):
    if dType_str=="float":
        return 4
    elif dType_str=="float16":
        return 2
    elif dType_str=="bf16":
        return 2


def gen_tiling(N, C, HXW, G, dtype_str):
# 声明变量
    BLOCK_BYTES = 32
    NXG = N * G
    C_G = C // G
    task_num_per_core = 0
    task_num_per_tail_core = 0
    tail_core = 0
    mode1_ub_cap_C_num = 0
    mode1_ub_iter_C_num = 0
    mode1_ub_tail_C_num = 0
    mode2_ub_capacity_ele = 0
    mode2_ub_iteration_num = 0
    mode2_ub_tail_num = 0
    dx_is_require = 1
    dgamma_is_require = 1
    dbeta_is_require = 1
    Tiling_key = -1
    RESERVE_SAPCE = 1024
    UB_size = 192 * 1024
    dtype_bytes = GetDataTypeSize(dtype_str)
    ele_per_block = BLOCK_BYTES // dtype_bytes
    core_num = 48
    RESERVE_SAPCE += ceil(C_G, ele_per_block) * dtype_bytes * 3 + 32 * 2
    mode0_ub_cap_G_num = (UB_size - RESERVE_SAPCE) // (ceil(C_G * HXW, ele_per_block) * dtype_bytes * 3)
    task_num_per_core = ceil(ceil(NXG, core_num) // core_num, mode0_ub_cap_G_num) if(mode0_ub_cap_G_num * core_num < NXG and mode0_ub_cap_G_num > 1) else ceil(NXG, core_num) // core_num
    core_num_used = (NXG - 1) // task_num_per_core + 1
    task_num_per_tail_core = task_num_per_core
    tail_core = core_num_used

    if (NXG % core_num_used != 0):
        task_num_per_tail_core = task_num_per_core - 1
        tail_core = NXG % core_num_used
    if mode0_ub_cap_G_num >= 1:
        Tiling_key = 0
    elif mode0_ub_cap_G_num == 0:
        Tiling_key = 1
        mode1_ub_cap_C_num = (UB_size - RESERVE_SAPCE) // (ceil(HXW, ele_per_block) * dtype_bytes * 3)
        mode1_ub_iter_C_num = ceil(C_G , mode1_ub_cap_C_num) // mode1_ub_cap_C_num
        mode1_ub_tail_C_num = mode1_ub_cap_C_num if(mode1_ub_iter_C_num * mode1_ub_cap_C_num - C_G == 0) else (C_G - ((mode1_ub_iter_C_num - 1) * mode1_ub_cap_C_num))

    if mode1_ub_cap_C_num == 0:
        Tiling_key = 2
        mode2_ub_capacity_ele = floor((UB_size - RESERVE_SAPCE) // (dtype_bytes * 3), ele_per_block)
        mode2_ub_iteration_num = ceil(HXW, mode2_ub_capacity_ele) // mode2_ub_capacity_ele
        mode2_ub_tail_num = mode2_ub_capacity_ele if (HXW - mode2_ub_iteration_num * mode2_ub_capacity_ele == 0) else (HXW - (mode2_ub_iteration_num - 1) * mode2_ub_capacity_ele)


    # 创建字典
    variables_dict = {
        "Tiling_key": Tiling_key,
        "N": N,
        "C": C,
        "HXW": HXW,
        "G": G,
        "NXG": NXG,
        "C_G": C_G,
        "task_num_per_core": task_num_per_core,
        "task_num_per_tail_core": task_num_per_tail_core,
        "tail_core": tail_core,
        "mode1_ub_cap_C_num": mode1_ub_cap_C_num,
        "mode1_ub_iter_C_num": mode1_ub_iter_C_num,
        "mode1_ub_tail_C_num": mode1_ub_tail_C_num,
        "mode2_ub_capacity_ele": mode2_ub_capacity_ele,
        "mode2_ub_iteration_num": mode2_ub_iteration_num,
        "mode2_ub_tail_num": mode2_ub_tail_num,
        "dx_is_require": dx_is_require,
        "dgamma_is_require": dgamma_is_require,
        "dbeta_is_require": dbeta_is_require
    }

    # 创建数组
    variables_array = [variables_dict[key] for key in variables_dict]

    # 打印结果
    print("tiling_data:",variables_array)
    return variables_array
def ceil(a, b):
    return ((a - 1) // b + 1) * b


def floor(a, b):
    return a // b * b

def main():
    params_list = gen_tiling(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5])  # python gen_tiling.py case0  sys.argv[1]="case0"
    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
