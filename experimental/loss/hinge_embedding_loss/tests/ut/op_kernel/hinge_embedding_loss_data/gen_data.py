# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Golden-data generator for HingeEmbeddingLoss."""

import numpy as np
import argparse
from pathlib import Path


def golden(input_data, target, margin, reduction):
    loss = np.where(target == 1.0, input_data, np.maximum(0.0, margin - input_data))
    if reduction == "sum":
        return np.array([loss.sum()], dtype=np.float32)
    if reduction == "mean":
        return np.array([loss.mean()], dtype=np.float32)
    return loss.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--reduction", choices=("none", "sum", "mean"), default="none")
    args = parser.parse_args()
    input_data = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0], dtype=np.float32)
    target = np.array([1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0], dtype=np.float32)
    output = golden(input_data, target, args.margin, args.reduction)
    output_dir = Path(__file__).resolve().parent
    input_data.tofile(output_dir / "input.bin")
    target.tofile(output_dir / "target.bin")
    output.tofile(output_dir / "loss_golden.bin")


if __name__ == "__main__":
    main()
