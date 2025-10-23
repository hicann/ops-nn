#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import numpy as np
import sys


def gen_tiling(batch, num_heads, channels, num_levels, num_points, num_queries):
    reserve_space = 1024
    ub_size = 192 * 1024
    core_num = 48
    ub_size = ub_size - reserve_space
    spatial_size = num_levels * 24
    task_per_core = (batch * num_queries - 1) // core_num + 1
    core_used = (batch * num_queries - 1) // task_per_core + 1
    task_tail_core = batch * num_queries - (core_used - 1) * task_per_core
    data_align = 8
    NUM_EMEBDDIM_BUFFER = 8
    NUM_QUERIE_BUFFER = 15
    NUM_CHANNEL_BUFFER = 13
    num_levels_align = (num_levels + data_align - 1) // data_align * data_align
    max_ub_num = (ub_size // 4 - 3 * num_levels_align - NUM_EMEBDDIM_BUFFER * channels) // (NUM_QUERIE_BUFFER + NUM_CHANNEL_BUFFER * channels)
    max_ub_num = max_ub_num // data_align * data_align
    variables_dict = {
        "batch_size": batch,
        "spatial_size": spatial_size,
        "num_heads": num_heads,
        "channels": channels,
        "num_levels": num_levels,
        "num_query": num_queries,
        "num_point": num_points,
        "max_ub_num": max_ub_num
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))  # python gen_tiling.py
    base_params = np.array(params_list, dtype=np.uint64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
