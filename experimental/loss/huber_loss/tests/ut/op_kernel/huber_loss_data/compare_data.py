# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np


def read_tensor(path, dtype):
    if dtype == "bfloat16":
        values = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
        return values.view(np.float32)
    return np.fromfile(path, dtype=dtype).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Compare HuberLoss output with golden data."
    )
    parser.add_argument("dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    golden_path = args.golden
    if golden_path is None:
        golden_path = (
            Path(__file__).resolve().parent / f"{args.dtype}_golden_huber_loss.bin"
        )

    output = read_tensor(args.output, args.dtype)
    golden = read_tensor(golden_path, args.dtype)
    tolerance = (
        1e-5 if args.dtype == "float32" else (2e-2 if args.dtype == "float16" else 8e-2)
    )
    if output.shape != golden.shape:
        print(f"FAILED: output size {output.size} != golden size {golden.size}")
        return 1
    mismatches = np.flatnonzero(
        ~np.isclose(output, golden, rtol=tolerance, atol=tolerance, equal_nan=True)
    )
    if mismatches.size == 0:
        print("PASSED")
        return 0
    print(f"FAILED: {mismatches.size} mismatches")
    for index in mismatches[:10]:
        print(f"index={index}, output={output[index]}, golden={golden[index]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
