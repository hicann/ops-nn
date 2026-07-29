# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
"""Compare kernel output (out_var.bin) with the golden (golden.bin) for apply_gradient_descent.

Usage: python3 compare_data.py <dtype>   where <dtype> in {float16, float32, bfloat16}
Exit code 0 on pass, 1 on mismatch.
"""

import sys
import numpy as np
import ml_dtypes

DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": ml_dtypes.bfloat16,
}

# rtol/atol per dtype (aligned with ops precision standards for elementwise ops).
TOL_MAP = {
    "float16": (1e-3, 1e-3),
    "float32": (1e-5, 1e-6),
    "bfloat16": (4e-3, 4e-3),
}


def main():
    dtype_str = sys.argv[1] if len(sys.argv) > 1 else "float32"
    np_dtype = DTYPE_MAP[dtype_str]
    rtol, atol = TOL_MAP[dtype_str]

    out = np.fromfile("out_var.bin", dtype=np_dtype).astype(np.float32)
    golden = np.fromfile("golden.bin", dtype=np_dtype).astype(np.float32)

    if out.shape != golden.shape:
        print("[compare] shape mismatch: out=%s golden=%s" % (out.shape, golden.shape))
        sys.exit(1)

    diff = np.abs(out - golden)
    tol = atol + rtol * np.abs(golden)
    fail_mask = diff > tol
    fail_cnt = int(np.sum(fail_mask))
    max_err = float(np.max(diff)) if diff.size else 0.0
    print(
        "[compare] dtype=%s count=%d max_abs_err=%.6g fail_cnt=%d (rtol=%g atol=%g)"
        % (dtype_str, out.size, max_err, fail_cnt, rtol, atol)
    )

    if fail_cnt > 0:
        idx = np.argmax(diff)
        print(
            "[compare] worst idx=%d out=%.6g golden=%.6g" % (idx, out[idx], golden[idx])
        )
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
