# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse

import numpy as np


def golden(input_data, target, var, full=False, eps=1e-6, reduction="none"):
    """Compute GaussianNllLoss in float32 with NumPy broadcasting."""
    variance = np.maximum(var.astype(np.float32), np.float32(eps))
    difference = input_data.astype(np.float32) - target.astype(np.float32)
    loss = 0.5 * (np.log(variance) + difference * difference / variance)
    if full:
        loss += np.float32(0.5 * np.log(2.0 * np.pi))
    if reduction == "sum":
        return np.array([loss.sum()], dtype=np.float32)
    if reduction == "mean":
        return np.array([loss.mean()], dtype=np.float32)
    return loss.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--reduction", choices=("none", "sum", "mean"), default="none")
    args = parser.parse_args()

    input_data = np.linspace(-1.5, 2.0, 24, dtype=np.float32).reshape(2, 3, 4)
    target = np.array(
        [[[0.0, 0.25, 0.5, 0.75]], [[-0.5, -0.25, 0.0, 0.25]]],
        dtype=np.float32,
    )
    var = np.array([[0.0, 0.5, 1.0], [1.5, 2.0, 0.25]], dtype=np.float32)[..., None]
    output = golden(input_data, target, var, args.full, args.eps, args.reduction)
    input_data.tofile("input.bin")
    target.tofile("target.bin")
    var.tofile("var.bin")
    output.tofile("loss_golden.bin")


if __name__ == "__main__":
    main()
