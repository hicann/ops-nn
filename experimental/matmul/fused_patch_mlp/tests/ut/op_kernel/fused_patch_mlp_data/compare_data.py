#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
from pathlib import Path

import numpy as np


def unpack_bf16(values):
    values = np.asarray(values, dtype=np.uint16)
    return (values.astype(np.uint32) << 16).view(np.float32)


def load_tensor(path, dtype):
    if dtype == "float16":
        return np.fromfile(path, np.float16).astype(np.float32)
    if dtype == "bfloat16":
        return unpack_bf16(np.fromfile(path, np.uint16))
    return np.fromfile(path, np.float32)


def compare(dtype):
    root = Path(__file__).resolve().parent
    golden_files = sorted(root.glob(f"{dtype}_golden_*.bin"))
    output_files = sorted(root.glob(f"{dtype}_output_*.bin"))
    if not golden_files or len(golden_files) != len(output_files):
        print(f"FAILED! golden={golden_files}, output={output_files}")
        return False

    tolerance = {"float16": 1.0e-2, "bfloat16": 4.0e-2, "float32": 1.0e-4}[dtype]
    success = True
    for golden_path, output_path in zip(golden_files, output_files):
        golden = load_tensor(golden_path, dtype)
        output = load_tensor(output_path, dtype)
        if output.size != golden.size:
            print(f"FAILED! size mismatch: output={output.size}, golden={golden.size}")
            success = False
            continue

        close = np.isclose(
            output, golden, rtol=tolerance, atol=tolerance, equal_nan=True
        )
        bad_indices = np.flatnonzero(~close)
        if bad_indices.size == 0:
            print("PASSED!")
            continue

        print(f"FAILED! mismatches={bad_indices.size}/{golden.size}")
        for index in bad_indices[:5]:
            print(f"index: {index}, output: {output[index]}, golden: {golden[index]}")
        success = False
    return success


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("float16", "bfloat16", "float32"):
        raise SystemExit(f"usage: {sys.argv[0]} DTYPE")
    raise SystemExit(0 if compare(sys.argv[1]) else 1)


if __name__ == "__main__":
    main()
