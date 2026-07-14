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
from datetime import datetime

curr_dir = os.path.dirname(os.path.realpath(__file__))

OP_NAME = "relu_grad_v3"
OUTPUT_NAMES = ["z"]

TOLERANCE = {
    "float32": {"np_dtype": np.float32, "rtol": 1e-4, "atol": 1e-4},
    "float16": {"np_dtype": np.float16, "rtol": 1e-2, "atol": 1e-3},
    "bfloat16": {"np_dtype": "bfloat16", "rtol": 2e-2, "atol": 2e-3},
    "int32": {"np_dtype": np.int32, "rtol": 0, "atol": 0},
    "uint8": {"np_dtype": np.uint8, "rtol": 0, "atol": 0},
    "int8": {"np_dtype": np.int8, "rtol": 0, "atol": 0},
}


def unpack_bf16(data_u16):
    data_u32 = data_u16.astype(np.uint32) << 16
    return data_u32.view(np.float32)


def load_bin(path, np_dtype):
    if np_dtype == "bfloat16":
        return unpack_bf16(np.fromfile(path, np.uint16))
    return np.fromfile(path, np_dtype)


def get_file_pairs(dtype):
    pairs = []
    for name in OUTPUT_NAMES:
        golden = os.path.join(curr_dir, f"{dtype}_golden_{name}_t_{OP_NAME}.bin")
        output = os.path.join(curr_dir, f"{dtype}_output_{name}_t_{OP_NAME}.bin")
        pairs.append((golden, output, name))
    return pairs


def compare_one_pair(golden_path, output_path, label, np_dtype, rtol, atol):
    lines = []
    for path, role in [(golden_path, "golden"), (output_path, "output")]:
        if not os.path.exists(path):
            lines.append(f"[SKIP] {label}: {role} file not found: {path}")
            return False, lines

    golden_data = load_bin(golden_path, np_dtype)
    output_data = load_bin(output_path, np_dtype)

    if golden_data.shape != output_data.shape:
        lines.append(
            f"[FAIL] {label}: shape mismatch "
            f"(golden={golden_data.shape}, output={output_data.shape})"
        )
        return False, lines

    nelem = golden_data.shape[0]
    diff_mask = ~np.isclose(output_data, golden_data, rtol, atol, equal_nan=True)
    diff_count = int(np.sum(diff_mask))

    if diff_count == 0:
        lines.append(f"[PASS] {label} ({nelem} elements, 0 mismatches)")
        return True, lines

    lines.append(f"[FAIL] {label} ({nelem} elements, {diff_count} mismatches)")
    lines.append("  First 10 mismatches:")
    diff_indices = np.where(diff_mask)[0]
    for idx in diff_indices[:10]:
        lines.append(
            f"    idx={idx:<8} output={float(output_data[idx]):.6e}  "
            f"golden={float(golden_data[idx]):.6e}  "
            f"diff={abs(float(output_data[idx]) - float(golden_data[idx])):.6e}"
        )
    return False, lines


def write_log(dtype, tol_cfg, results, all_lines):
    log_path = os.path.join(curr_dir, f"compare_result_{dtype}.log")
    passed = sum(1 for ok, _ in results if ok)
    failed = len(results) - passed

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 64 + "\n")
        f.write(f"{OP_NAME} UT Compare Report\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dtype: {dtype}\n")
        f.write(f"Tolerance: rtol={tol_cfg['rtol']}, atol={tol_cfg['atol']}\n")
        f.write("=" * 64 + "\n\n")
        for lines in all_lines:
            for line in lines:
                f.write(line + "\n")
            f.write("\n")
        f.write(f"Result: {passed} PASSED, {failed} FAILED\n")

    for lines in all_lines:
        for line in lines:
            print(line)
    print(f"Result: {passed} PASSED, {failed} FAILED")
    print(f"Log saved to: {log_path}")


def process(dtype):
    if dtype not in TOLERANCE:
        raise ValueError(
            f"Unsupported dtype: {dtype}, must be one of {list(TOLERANCE.keys())}"
        )
    tol_cfg = TOLERANCE[dtype]
    pairs = get_file_pairs(dtype)
    if not pairs:
        print("[ERROR] No output names configured")
        return False
    results, all_lines = [], []
    for golden_path, output_path, label in pairs:
        ok, lines = compare_one_pair(
            golden_path,
            output_path,
            label,
            tol_cfg["np_dtype"],
            tol_cfg["rtol"],
            tol_cfg["atol"],
        )
        results.append((ok, label))
        all_lines.append(lines)
    write_log(dtype, tol_cfg, results, all_lines)
    return all(ok for ok, _ in results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: compare_data.py <dtype>")
        print(f"  Supported dtypes: {list(TOLERANCE.keys())}")
        exit(1)
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
