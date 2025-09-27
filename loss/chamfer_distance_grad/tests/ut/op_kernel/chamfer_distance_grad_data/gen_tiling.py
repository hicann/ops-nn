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
import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(current_path)
input_path = os.path.join(project_path, "build/input")


def gen_tiling(batch, num):
    reserve_space = 16 * 1024
    ub_size = 192 * 1024
    core_num = 48
    ub_size = ub_size - reserve_space
    task_per_core = (batch * num - 1) // core_num + 1
    core_used = (batch * num - 1) // task_per_core + 1
    task_tail_core = batch * num - (core_used - 1) * task_per_core
    variables_dict = {
        "batch_size": batch,
        "num": num,
        "ub_size": ub_size,
        "task_per_core": task_per_core,
        "core_used": core_used,
        "task_tail_core": task_tail_core
    }

    variables_array = [variables_dict[key] for key in variables_dict]
    print("tiling_data:", variables_array)
    return variables_array


def main():
    params_list = gen_tiling(int(sys.argv[1]), int(sys.argv[2]))  # python gen_tiling.py
    base_params = np.array(params_list, dtype=np.uint32)
    base_params.tofile(os.path.join(input_path, "tiling.bin"))


if __name__ == '__main__':
    main()
