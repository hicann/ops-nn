#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np


def main():
    shape = tuple(int(value) for value in sys.argv[1].strip("()").split(",") if value)
    alpha = float(sys.argv[2])
    beta = float(sys.argv[3])
    values = np.array(
        [
            -np.inf,
            -8.0,
            -4.0,
            -3.0,
            -2.999,
            -1.0,
            0.0,
            1.0,
            2.999,
            3.0,
            4.0,
            8.0,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    count = int(np.prod(shape))
    x = np.resize(values, count).reshape(shape)
    golden = np.clip(alpha * x + beta, 0.0, 1.0).astype(np.float32)
    x.tofile("input_x.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    main()
