#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden comparator for QuantizeAddLayerNorm kernel UT.

Usage:
    python3 compare_data.py 'float32' <mode>     # mode in {mul, div, per_tensor}

Compares:
    output_y.bin  vs golden_y_<mode>.bin   (int8: allow +-1 rounding slack on <=2% elements)
    output_x.bin  vs golden_x.bin          (fp32: tight tolerance)
"""

import os
import sys
import numpy as np

INT8_SLACK_RATIO = 0.02


def load(path, dtype):
    if not os.path.exists(path):
        print("[FAIL] missing file: {}".format(path))
        sys.exit(1)
    return np.fromfile(path, dtype=dtype)


def compare_int8(out_path, golden_path):
    out = load(out_path, np.int8).astype(np.int32)
    golden = load(golden_path, np.int8).astype(np.int32)
    if out.shape != golden.shape:
        print(
            "[FAIL] y shape mismatch: out={} golden={}".format(out.shape, golden.shape)
        )
        return False
    diff = np.abs(out - golden)
    hard = int(np.count_nonzero(diff > 1))
    soft = int(np.count_nonzero(diff == 1))
    total = out.size
    print(
        "y(int8) total={} exact={} off_by_1={} off_by_>1={}".format(
            total, total - soft - hard, soft, hard
        )
    )
    if hard > 0:
        idx = int(np.argmax(diff > 1))
        print(
            "[FAIL] y hard mismatch at [{}] out={} golden={}".format(
                idx, out[idx], golden[idx]
            )
        )
        return False
    if total > 0 and soft / total > INT8_SLACK_RATIO:
        print(
            "[FAIL] y off-by-1 ratio {:.4%} exceeds {:.0%}".format(
                soft / total, INT8_SLACK_RATIO
            )
        )
        return False
    return True


def compare_float(out_path, golden_path):
    out = load(out_path, np.float32)
    golden = load(golden_path, np.float32)
    if out.shape != golden.shape:
        print(
            "[FAIL] x shape mismatch: out={} golden={}".format(out.shape, golden.shape)
        )
        return False
    bad = np.where(~np.isclose(out, golden, rtol=1e-5, atol=1e-6))[0]
    total = out.size
    print(
        "x(fp32) total={} ok={} mismatch={}".format(total, total - bad.size, bad.size)
    )
    if bad.size > 0:
        idx = int(bad[0])
        print(
            "[FAIL] x first mismatch at [{}] out={} golden={}".format(
                idx, out[idx], golden[idx]
            )
        )
        return False
    return True


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    d_type = sys.argv[1]
    mode = sys.argv[2]
    assert mode in ("mul", "div", "per_tensor"), "unknown mode: {}".format(mode)
    prefix = "{}_".format(d_type)

    ok = True
    ok &= compare_int8(prefix + "output_y.bin", prefix + "golden_y_{}.bin".format(mode))
    ok &= compare_float(prefix + "output_x.bin", prefix + "golden_x.bin")

    if ok:
        print("[PASS] mode={}".format(mode))
        sys.exit(0)
    print("[FAIL] mode={}".format(mode))
    sys.exit(1)


if __name__ == "__main__":
    main()
