# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import sys

import numpy as np


def compare(actual_path, golden_path, rtol, atol, name):
    actual = np.fromfile(actual_path, dtype=np.float32)
    golden = np.fromfile(golden_path, dtype=np.float32)
    if actual.shape != golden.shape:
        print(f"{name} shape mismatch: actual={actual.shape}, golden={golden.shape}")
        return False
    maximum_error = np.max(np.abs(actual - golden)) if actual.size else 0.0
    print(f"{name} maximum absolute error: {maximum_error:.9g}")
    return bool(np.allclose(actual, golden, rtol=rtol, atol=atol))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actual_grad_input")
    parser.add_argument("golden_grad_input")
    parser.add_argument("actual_grad_var")
    parser.add_argument("golden_grad_var")
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()
    passed = compare(
        args.actual_grad_input,
        args.golden_grad_input,
        args.rtol,
        args.atol,
        "gradInput",
    )
    passed = (
        compare(
            args.actual_grad_var, args.golden_grad_var, args.rtol, args.atol, "gradVar"
        )
        and passed
    )
    print(
        "GaussianNllLossGrad comparison passed"
        if passed
        else "GaussianNllLossGrad comparison failed"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
