# This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
# and is contributed to the CANN Open Software.

# Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
# All Rights Reserved.

# Authors (accounts):
# - Pei Haobo<@xiaopei-1>
# - Su Tonghua <@sutonghua>

# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import os
import numpy as np
import glob

curr_dir = os.path.dirname(os.path.realpath(__file__))


def unpack_bf16(data_u16):
    """将 bfloat16 (uint16) 还原为 float32。"""
    data_u32 = data_u16.astype(np.uint32) << 16
    return data_u32.view(np.float32)


def compare_data(golden_file_lists, output_file_lists, d_type):
    if d_type == "float16":
        np_dtype = np.float16
    elif d_type == "float32":
        np_dtype = np.float32
    elif d_type == "bfloat16":
        np_dtype = "bfloat16"
    else:
        raise ValueError("d_type must be float16, float32, or bfloat16")

    log_path = os.path.join(curr_dir, "compare_result.log")
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        if np_dtype == "bfloat16":
            tmp_out = unpack_bf16(np.fromfile(out, np.uint16))
            tmp_gold = unpack_bf16(np.fromfile(gold, np.uint16))
        else:
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)

        diff_res = np.isclose(tmp_out, tmp_gold, 1e-2, 1e-3, True)
        diff_idx = np.where(diff_res != True)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            msg = "FAILED! total mismatches: {}\n".format(len(diff_idx))
            for idx in diff_idx[:10]:
                msg += "index: {}, output: {}, golden: {}, diff: {}\n".format(
                    idx, tmp_out[idx], tmp_gold[idx],
                    abs(float(tmp_out[idx]) - float(tmp_gold[idx]))
                )
            print(msg)
            with open(log_path, "w") as f:
                f.write(msg)
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result


if __name__ == "__main__":
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
