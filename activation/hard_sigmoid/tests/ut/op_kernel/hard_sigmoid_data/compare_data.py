#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import numpy as np


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    diagnostic_path = os.path.join(current_dir, "compare_failure.txt")
    if os.path.exists(diagnostic_path):
        os.remove(diagnostic_path)
    golden = np.fromfile(os.path.join(current_dir, "golden.bin"), dtype=np.float32)
    output = np.fromfile(os.path.join(current_dir, "output.bin"), dtype=np.float32)
    if golden.shape != output.shape:
        message = f"shape mismatch: output={output.shape}, golden={golden.shape}"
        with open(diagnostic_path, "w", encoding="utf-8") as diagnostic:
            diagnostic.write(message + "\n")
        print(message)
        sys.exit(1)
    if not np.allclose(output, golden, rtol=1e-4, atol=1e-4, equal_nan=True):
        mismatch = np.flatnonzero(
            ~np.isclose(output, golden, rtol=1e-4, atol=1e-4, equal_nan=True)
        )
        if mismatch.size:
            index = int(mismatch[0])
            message = (
                f"mismatch_count={mismatch.size}; first mismatch at {index}: "
                f"output={output[index]}, golden={golden[index]}"
            )
            with open(diagnostic_path, "w", encoding="utf-8") as diagnostic:
                diagnostic.write(message + "\n")
            print(message)
        sys.exit(1)
    print("COMPARE DATA PASSED")


if __name__ == "__main__":
    main()
