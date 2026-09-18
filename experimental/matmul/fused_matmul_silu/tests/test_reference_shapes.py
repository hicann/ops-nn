#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Iterable, Tuple

import numpy as np


REFERENCE_SHAPES: Iterable[Tuple[int, int, int]] = (
    (2, 256, 4096),
    (2, 4096, 4096),
    (2, 4096, 2048),
)
DATA_STDDEV = 0.5


def to_bf16_bits(array: np.ndarray) -> np.ndarray:
    data = np.asarray(array, dtype=np.float32)
    bits = data.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return ((bits + rounding) >> 16).astype(np.uint16)


def from_bf16_bits(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16_round(array: np.ndarray) -> np.ndarray:
    return from_bf16_bits(to_bf16_bits(array))


def fused_matmul_silu_reference(m: int, k: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, DATA_STDDEV, size=(m, k)).astype(np.float32)
    weight = rng.normal(0.0, DATA_STDDEV, size=(n, k)).astype(np.float32)
    bias = rng.normal(0.0, DATA_STDDEV, size=(n,)).astype(np.float32)

    x_bf16 = bf16_round(x)
    weight_bf16 = bf16_round(weight)
    bias_bf16 = bf16_round(bias)
    matmul_bf16 = bf16_round(x_bf16 @ weight_bf16.T)
    linear = matmul_bf16 + bias_bf16
    silu = linear / (1.0 + np.exp(-linear))
    return bf16_round(silu)


def main() -> None:
    for index, (m, k, n) in enumerate(REFERENCE_SHAPES):
        y = fused_matmul_silu_reference(m, k, n, seed=2026 + index)
        assert y.shape == (m, n)
        assert np.isfinite(y).all()
        print(
            f"shape={(m, k, n)} ok, min={float(y.min()):.6f}, "
            f"max={float(y.max()):.6f}, mean={float(y.mean()):.6f}"
        )
    print("reference shape self-check passed")


if __name__ == "__main__":
    main()
