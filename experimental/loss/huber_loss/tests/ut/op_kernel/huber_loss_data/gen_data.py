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


def parse_shape(value):
    dimensions = [int(item) for item in value.strip("()[]").split(",") if item.strip()]
    return tuple(dimensions)


def pack_bfloat16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16).astype(np.uint16)


def huber_loss(predictions, targets, delta):
    difference = predictions - targets
    absolute = np.abs(difference)
    return np.where(
        absolute <= delta,
        0.5 * difference * difference,
        delta * (absolute - 0.5 * delta),
    )


def write_tensor(path, values, dtype):
    if dtype == "bfloat16":
        pack_bfloat16(values).tofile(path)
    else:
        values.astype(dtype).tofile(path)


def main():
    parser = argparse.ArgumentParser(description="Generate HuberLoss kernel test data.")
    parser.add_argument("shape", help="Tensor shape, for example '(2,3)'")
    parser.add_argument("dtype", choices=("float16", "float32", "bfloat16"))
    parser.add_argument("--delta", type=float, default=1.0)
    args = parser.parse_args()
    if args.delta <= 0:
        raise ValueError("delta must be greater than 0")

    shape = parse_shape(args.shape)
    size = int(np.prod(shape, dtype=np.int64))
    values = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    predictions = np.resize(values, size).reshape(shape)
    targets = np.resize(values[::-1], size).reshape(shape)
    golden = huber_loss(predictions, targets, args.delta)
    output_dir = Path(__file__).resolve().parent
    write_tensor(
        output_dir / f"{args.dtype}_predictions_huber_loss.bin", predictions, args.dtype
    )
    write_tensor(
        output_dir / f"{args.dtype}_targets_huber_loss.bin", targets, args.dtype
    )
    write_tensor(output_dir / f"{args.dtype}_golden_huber_loss.bin", golden, args.dtype)


if __name__ == "__main__":
    main()
