# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reference implementation of TurboQuantCompressLatent.

The operation order mirrors the kernel exactly (reduce -> reciprocal -> scale -> midpoint counting), so the
output can be compared byte for byte instead of with a tolerance.
"""

import math

import numpy as np

HEAD_DIM = 512
N_CENT = 16
SCALE_BYTES = 2
SLOT_ALIGN = 64
NORM_EPS = np.float32(1e-16)
OUTPUT_COMPACT_CORRECTED = 1


def slot_size(head_dim, output_mode=0):
    used = head_dim // 2 + SCALE_BYTES
    if output_mode == OUTPUT_COMPACT_CORRECTED:
        return used
    return (used + SLOT_ALIGN - 1) // SLOT_ALIGN * SLOT_ALIGN


def make_centroids(head_dim=HEAD_DIM):
    """Lloyd-Max codebook for N(0, 1/head_dim), the codebook TurboQuant uses in production."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400000) * (1.0 / math.sqrt(head_dim))
    c = np.linspace(-2 / math.sqrt(head_dim), 2 / math.sqrt(head_dim), N_CENT)
    for _ in range(60):
        a = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        n = np.array(
            [x[a == i].mean() if np.any(a == i) else c[i] for i in range(N_CENT)]
        )
        if np.allclose(n, c):
            break
        c = n
    return np.sort(c).astype(np.float32)


def golden_nibbles(latent, centroids):
    latent = np.ascontiguousarray(latent, dtype=np.float32)
    red = np.sum(latent * latent, axis=1, dtype=np.float32)
    norm = np.sqrt(red + NORM_EPS).astype(np.float32)
    inv = (np.float32(1.0) / norm).astype(np.float32)
    unit = (latent * inv[:, None]).astype(np.float32)
    bnd = ((centroids[:-1] + centroids[1:]) * np.float32(0.5)).astype(np.float32)
    # counting how many midpoint boundaries a value exceeds == index of the nearest centroid
    nib = (unit[:, :, None] >= bnd[None, None, :]).sum(axis=2).astype(np.uint8)
    return nib, norm


def golden_compress(latent, centroids, output_mode=0):
    latent = np.ascontiguousarray(latent, dtype=np.float32)
    num_tokens, head_dim = latent.shape
    packed = head_dim // 2
    nib, norm = golden_nibbles(latent, centroids)

    out = np.zeros((num_tokens, slot_size(head_dim, output_mode)), dtype=np.uint8)
    out[:, :packed] = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    scale = norm
    if output_mode == OUTPUT_COMPACT_CORRECTED:
        selected = centroids[nib]
        centroid_norm = np.sqrt(
            np.sum(selected * selected, axis=1, dtype=np.float32) + NORM_EPS
        ).astype(np.float32)
        scale = (norm / centroid_norm).astype(np.float32)
    out[:, packed : packed + SCALE_BYTES] = (
        scale.astype(np.float16).view(np.uint8).reshape(num_tokens, SCALE_BYTES)
    )
    return out
