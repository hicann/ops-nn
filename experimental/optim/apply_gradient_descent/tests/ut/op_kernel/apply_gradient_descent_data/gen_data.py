# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

#!/usr/bin/python3
"""Generate input (var/alpha/delta) and golden data for the apply_gradient_descent kernel UT.

Semantics: var_out = var - alpha * delta, computed elementwise in fp32.
Usage: python3 gen_data.py <dtype>   where <dtype> in {float16, float32, bfloat16}
"""

import sys
import numpy as np
import ml_dtypes

DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "bfloat16": ml_dtypes.bfloat16,
}

SHAPE = (30, 4, 2)


def main():
    dtype_str = sys.argv[1] if len(sys.argv) > 1 else "float32"
    np_dtype = DTYPE_MAP[dtype_str]
    np.random.seed(2026)

    total = int(np.prod(SHAPE))
    var_f32 = np.random.uniform(-2.0, 2.0, total).astype(np.float32)
    delta_f32 = np.random.uniform(-1.0, 1.0, total).astype(np.float32)
    alpha_f32 = np.array([0.1], dtype=np.float32)

    # Round the inputs into the target dtype (this is what the kernel actually reads).
    var = var_f32.astype(np_dtype)
    delta = delta_f32.astype(np_dtype)
    alpha = alpha_f32.astype(np_dtype)

    # Golden is computed in fp32 from the target-dtype-rounded inputs, then cast back.
    var_up = var.astype(np.float32)
    delta_up = delta.astype(np.float32)
    alpha_up = alpha.astype(np.float32)
    golden_f32 = var_up - alpha_up[0] * delta_up
    golden = golden_f32.astype(np_dtype)

    var.tofile("var.bin")
    alpha.tofile("alpha.bin")
    delta.tofile("delta.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    main()
