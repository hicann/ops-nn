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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("actual")
    parser.add_argument("golden")
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()
    actual = np.fromfile(args.actual, dtype=np.float32)
    golden = np.fromfile(args.golden, dtype=np.float32)
    if actual.shape != golden.shape or not np.allclose(
        actual, golden, rtol=args.rtol, atol=args.atol
    ):
        max_error = (
            np.max(np.abs(actual - golden))
            if actual.shape == golden.shape and actual.size
            else np.inf
        )
        print(f"GaussianNllLoss comparison failed, max error: {max_error}")
        return 1
    print("GaussianNllLoss comparison passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
