#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Validate torch.ops.ascend_ops_nn.fast_hadamard against a CPU reference.

The kernel computes the UNNORMALIZED Hadamard transform: out[row] = H_n @ x[row],
where H_n is the n x n Sylvester (+-1) matrix (no 1/sqrt(n) scaling).
"""

import logging

import torch
import torch_npu
import ascend_ops_nn

torch_npu.__name__
ascend_ops_nn.__name__

logging.basicConfig(level=logging.INFO)


def hadamard_matrix(n, dtype=torch.float32):
    """n x n Sylvester Hadamard matrix (+-1), n a power of two."""
    h = torch.ones(1, 1, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def run_case(batch, n):
    x = torch.randn(batch, n, dtype=torch.float16)
    out = torch.empty_like(x)

    h = hadamard_matrix(n)
    ref = (x.float() @ h.t()).to(torch.float16)  # H symmetric: x @ H^T == H @ x rows

    x_npu = x.npu()
    out_npu = out.npu()
    torch.ops.ascend_ops_nn.fast_hadamard(x_npu, out_npu)

    got = out_npu.cpu().float()
    ref_f = ref.float()
    cos = torch.nn.functional.cosine_similarity(
        got.reshape(-1), ref_f.reshape(-1), dim=0
    ).item()
    max_abs = (got - ref_f).abs().max().item()
    logging.info(f"batch={batch} n={n}: cosine={cos:.6f} max_abs_err={max_abs:.4f}")
    assert cos > 0.999, f"cosine {cos} too low for batch={batch} n={n}"


if __name__ == "__main__":
    for n in [64, 128, 256, 512, 1024, 2048, 4096]:
        run_case(8, n)
    logging.info("fast_hadamard torch.ops test PASSED")
