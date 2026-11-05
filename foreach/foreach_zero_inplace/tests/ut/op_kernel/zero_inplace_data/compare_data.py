#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import glob
import os
import struct
import sys

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    formats = {
        "float16": ("e", 2),
        "float32": ("f", 4),
        "int16": ("h", 2),
        "int32": ("i", 4),
        "bfloat16_t": ("bfloat16", 2),
    }
    if d_type not in formats:
        raise ValueError(f"Unsupported dtype: {d_type}")
    value_format, item_size = formats[d_type]

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        with open(out, "rb") as output_file:
            output_data = output_file.read()
        with open(gold, "rb") as golden_file:
            golden_data = golden_file.read()
        if len(output_data) != len(golden_data) or len(output_data) % item_size != 0:
            print(f"FAILED! size mismatch: {out}")
            data_same = False
            continue

        diff_idx = []
        for index in range(0, len(golden_data), item_size):
            output_bytes = output_data[index : index + item_size]
            golden_bytes = golden_data[index : index + item_size]
            output_value = unpack_value(output_bytes, value_format)
            golden_value = unpack_value(golden_bytes, value_format)
            equal = output_bytes == golden_bytes
            if not equal:
                diff_idx.append((index // item_size, output_value, golden_value))

        if not diff_idx:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx, output_value, golden_value in diff_idx[:5]:
                print(f"index: {idx}, output: {output_value}, golden: {golden_value}")
            data_same = False
    return data_same


def unpack_value(data, value_format):
    if value_format == "bfloat16":
        bits = struct.unpack("<H", data)[0] << 16
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<" + value_format, data)[0]


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    if len(golden_file_lists) == 0 or len(golden_file_lists) != len(output_file_lists):
        print("FAILED! missing or incomplete output files")
        return False
    return compare_data(golden_file_lists, output_file_lists, d_type)


if __name__ == "__main__":
    sys.exit(0 if process(sys.argv[1]) else 1)
