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


def hinge_loss_golden(predict, target):
    return np.maximum(0.0, 1.0 - target * predict)


def pack_bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(
        np.uint16
    )


def generate(shape_text, dtype):
    shape = tuple(
        int(value.strip())
        for value in shape_text.strip("()").split(",")
        if value.strip()
    )
    size = int(np.prod(shape))
    predict = np.resize(
        np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32), size
    ).reshape(shape)
    target = np.resize(np.array([1.0, -1.0], dtype=np.float32), size).reshape(shape)
    golden = hinge_loss_golden(predict, target)
    if dtype == "bfloat16":
        pack_bf16(predict).tofile("bfloat16_predict_t_hinge_loss.bin")
        pack_bf16(target).tofile("bfloat16_target_t_hinge_loss.bin")
        pack_bf16(golden).tofile("bfloat16_golden_loss_t_hinge_loss.bin")
        return
    np_dtype = np.float16 if dtype == "float16" else np.float32
    predict.astype(np_dtype).tofile(f"{dtype}_predict_t_hinge_loss.bin")
    target.astype(np_dtype).tofile(f"{dtype}_target_t_hinge_loss.bin")
    golden.astype(np_dtype).tofile(f"{dtype}_golden_loss_t_hinge_loss.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: gen_data.py <shape> <float32|float16|bfloat16>")
    for name in os.listdir("."):
        if name.endswith(".bin"):
            os.remove(name)
    generate(sys.argv[1], sys.argv[2])
