#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Validate torch.ops.ascend_ops_nn.fast_hadamard_quant by EXACT bit-equality against a
reference (mirrors pto-kernels' test_hadamard_quant.py). The op is a deterministic
quantizer -- unnormalized Hadamard transform, then int4 quant q = clamp(round(x*scale +
q_offset), -8, 7), packed two-per-byte (even idx -> low nibble, odd idx -> high nibble).
The reference runs the same fp16 butterfly on-device, so the bytes match exactly.
"""

import logging
import math

import torch
import torch_npu
import ascend_ops_nn

torch_npu.__name__
ascend_ops_nn.__name__

logging.basicConfig(level=logging.INFO)
DTYPE = torch.float16


def hadamard_butterfly(x):
    """Unnormalized H_n . x via the same fp16 even/odd butterfly the kernel uses."""
    x = x.clone()
    n = x.shape[-1]
    n_half = n // 2
    for _ in range(int(math.log2(n))):
        even = x[..., 0::2].clone()
        odd = x[..., 1::2].clone()
        x[..., :n_half] = even + odd
        x[..., n_half:] = even - odd
    return x


def quantize_int4_values(x, scale, q_offset=0.0):
    s = torch.tensor(scale, device=x.device, dtype=DTYPE)
    o = torch.tensor(q_offset, device=x.device, dtype=DTYPE)
    q = torch.round((x * s + o).float()).to(torch.int32)
    return torch.clamp(q, -8, 7).to(torch.int8)


def pack_int4(q):
    """q int8 [batch, n] in [-8, 7] -> packed int8 [batch, n/2]."""
    q = q.to(torch.int32)
    low = torch.bitwise_and(q[:, 0::2], 0xF)
    high = torch.bitwise_and(q[:, 1::2], 0xF)
    packed = torch.bitwise_or(low, torch.bitwise_left_shift(high, 4)).to(torch.int16)
    packed = torch.where(packed >= 128, packed - 256, packed)
    return packed.to(torch.int8)


def run_case(batch, n, scale, q_offset=0.0):
    x = torch.randn(batch, n, dtype=DTYPE).npu()
    y = torch.empty(batch, n // 2, dtype=torch.int8).npu()

    y_ref = pack_int4(quantize_int4_values(hadamard_butterfly(x), scale, q_offset))

    # group_scales/group_offsets absent, group_size=0 -> uniform scale over the whole row.
    torch.ops.ascend_ops_nn.fast_hadamard_quant(x, None, None, scale, 0, q_offset, y)

    assert torch.equal(y, y_ref), (
        f"mismatch batch={batch} n={n} scale={scale} q_offset={q_offset}: "
        f"{(y != y_ref).sum().item()} of {y.numel()} bytes differ"
    )
    logging.info(f"batch={batch} n={n} scale={scale} q_offset={q_offset}: EXACT match")


if __name__ == "__main__":
    for n in [128, 256, 512, 1024]:
        for scale in [0.5, 1.0, 2.0]:
            run_case(7, n, scale)
    run_case(7, 1024, 1.25, q_offset=1.5)  # offset path
    logging.info("fast_hadamard_quant torch.ops test PASSED")
