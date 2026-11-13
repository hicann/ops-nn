# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))

OUTPUT_NUM = 4


def compare_data(golden_file_lists, output_file_lists):
    # int64 整数语义要求逐元素完全一致（binary_equal），不做容差比较
    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np.int64)
        tmp_gold = np.fromfile(gold, np.int64)
        if np.array_equal(tmp_gold, tmp_out):
            print(f"PASSED! {os.path.basename(out)}")
        else:
            print(f"FAILED! {os.path.basename(out)}")
            diff_idx = np.where(tmp_gold != tmp_out)[0]
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(prefix):
    # 4 个输出（inputTokens/inputPositions/seqLens/slotMapping）必须成对齐全：
    # 缺任一 golden 或 output 即失败——避免部分结果缺失时静默跳过造成假通过
    golden_file_lists = []
    output_file_lists = []
    missing = []
    for i in range(1, OUTPUT_NUM + 1):
        golden = os.path.join(curr_dir, f"{prefix}{i}_golden_advance_step.bin")
        output = os.path.join(curr_dir, f"{prefix}{i}_output_advance_step.bin")
        if not os.path.exists(golden):
            missing.append(os.path.basename(golden))
        if not os.path.exists(output):
            missing.append(os.path.basename(output))
        golden_file_lists.append(golden)
        output_file_lists.append(output)
    if missing:
        print(f"FAILED! missing files for prefix '{prefix}': {', '.join(missing)}")
        sys.exit(1)
    return golden_file_lists, output_file_lists


def process(prefix):
    golden_file_lists, output_file_lists = get_file_lists(prefix)
    if not compare_data(golden_file_lists, output_file_lists):
        sys.exit(1)


if __name__ == "__main__":
    # argv[1] 保留 dtype 兼容旧调用；argv[2] 可选文件前缀（V2 用例为 v2_）
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""
    process(prefix)
