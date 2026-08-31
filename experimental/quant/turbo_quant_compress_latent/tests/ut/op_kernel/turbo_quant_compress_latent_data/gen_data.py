# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Input generator for the op_kernel UT.

Usage: gen_data.py <num_tokens> [distribution]

The distributions follow the ecosystem precision standard: regular cases are split between a uniform
distribution over [-5, 5] and a normal distribution with mu in [-5, 5] and sigma in [0.1, 2].
`gauss_lat` is the in-distribution case (a latent that has been through the signed Hadamard is close to
N(0, 1/headDim), which is what the codebook was fitted to). The remaining names inject non-finite values.
"""

import sys

import numpy as np

from golden import HEAD_DIM, N_CENT, golden_compress, golden_nibbles, make_centroids

SPECIAL = {"nan": np.nan, "posinf": np.inf, "neginf": -np.inf}


def build_input(num_tokens, dist, rng, centroids=None):
    shape = (num_tokens, HEAD_DIM)
    if dist == "gauss_lat":
        return (rng.standard_normal(shape) / np.sqrt(HEAD_DIM)).astype(np.float32)
    if dist == "uniform":
        return rng.uniform(-5.0, 5.0, shape).astype(np.float32)
    if dist == "gauss_wide":
        mu = rng.uniform(-5.0, 5.0)
        sigma = rng.uniform(0.1, 2.0)
        return (rng.standard_normal(shape) * sigma + mu).astype(np.float32)
    if dist == "all_centroids":
        if centroids is None:
            raise ValueError("all_centroids requires the production centroids")
        unit = np.empty(HEAD_DIM, dtype=np.float32)
        unit[:N_CENT] = centroids
        tail_square = (
            np.float32(1.0) - np.sum(centroids * centroids, dtype=np.float32)
        ) / (HEAD_DIM - N_CENT)
        unit[N_CENT:] = np.sqrt(tail_square).astype(np.float32)
        latent = np.tile(unit, (num_tokens, 1))
        codes, _ = golden_nibbles(latent, centroids)
        assert np.array_equal(np.unique(codes), np.arange(N_CENT, dtype=np.uint8))
        return latent

    # non-finite cases: scatter the special value through an otherwise in-distribution tensor so that the
    # poisoned and the clean lanes of the same token are both exercised
    base = (rng.standard_normal(shape) / np.sqrt(HEAD_DIM)).astype(np.float32)
    if dist in SPECIAL:
        base[:, ::64] = np.float32(SPECIAL[dist])
        return base
    if dist == "mixinf":
        base[:, ::64] = np.float32(np.inf)
        base[:, 1::64] = np.float32(-np.inf)
        base[:, 2::64] = np.float32(np.nan)
        return base
    raise ValueError("unknown distribution: %s" % dist)


def main():
    num_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    dist = sys.argv[2] if len(sys.argv) > 2 else "gauss_lat"
    output_mode = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(num_tokens if num_tokens > 0 else 1)

    centroids = make_centroids()
    assert centroids.shape == (N_CENT,)
    latent = build_input(num_tokens, dist, rng, centroids)

    latent.tofile("input_latent.bin")
    centroids.tofile("input_centroids.bin")
    golden_compress(latent, centroids, output_mode).tofile("golden_slot.bin")


if __name__ == "__main__":
    main()
