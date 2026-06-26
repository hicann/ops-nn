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
"""Minimal per-call latency benchmark for the fast Hadamard ops, called via torch.ops.

Self-contained: the copy baseline is a plain torch device-to-device copy (no external
deps). One optional column, rotq_us, times the installed rotate_quant op (cube-matmul
rotate + int8 dynamic quant) through a small PyTorch ACLNN launcher; it shows '-'
unless rotate_quant is built+installed:
    bash build.sh --pkg --soc=ascend910b --ops=rotate_quant && ./build_out/*.run

Run on the NPU server, after installing the ascend_ops_nn extension:
    ASCEND_RT_VISIBLE_DEVICES=<free-id> python3 benchmark.py
"""

import os
from pathlib import Path

import torch
import torch_npu
import ascend_ops_nn

torch_npu.__name__
ascend_ops_nn.__name__

WARMUP, REPEATS, POOL = 10, 50, 8
BATCH = 8192
DIMS = [128, 256, 512, 1024, 2048]


def bench_us(call):
    """Mean us/launch over a rotating buffer pool, timed with NPU events."""
    for i in range(WARMUP):
        call(i)
    torch.npu.synchronize()
    start, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )
    start.record()
    for i in range(REPEATS):
        call(WARMUP + i)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / REPEATS * 1e3  # ms -> us


def pool(factory):
    return [factory() for _ in range(POOL)]


def hadamard_matrix(n, dev):
    """n x n Sylvester (+-1) Hadamard rotation matrix, fp16 on device (rotate_quant input)."""
    h = torch.ones(1, 1, dtype=torch.float16)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h.to(dev)


def load_rotate_quant():
    """JIT-build the optional rotate_quant baseline binding. Returns the op or None."""
    from torch.utils.cpp_extension import load

    ascend = (
        os.environ.get("ASCEND_TOOLKIT_HOME")
        or os.environ.get("ASCEND_HOME_PATH")
        or "/usr/local/Ascend/cann"
    )
    api = f"{ascend}/opp/vendors/custom_nn/op_api"
    try:
        src = Path(__file__).with_name("rotate_quant_binding.cc")
        load(
            name="rotate_baseline_ext",
            sources=[str(src)],
            extra_include_paths=[f"{ascend}/include", f"{api}/include"],
            extra_ldflags=[
                f"-L{ascend}/lib64",
                "-lascendcl",
                "-lnnopbase",
                "-lacl_op_compiler",
                f"-L{api}/lib",
                "-lcust_opapi",
                f"-Wl,-rpath,{api}/lib",
            ],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=False,
            is_python_module=False,
        )
        return torch.ops.rotate_baseline.rotate_quant
    except (ImportError, OSError, RuntimeError) as e:
        print(
            f"[INFO] rotate_quant baseline off ({type(e).__name__}); "
            f"build it with: build.sh --pkg --soc=ascend910b --ops=rotate_quant"
        )
        return None


def main():
    torch.npu.set_device(0)  # physical NPU chosen via ASCEND_RT_VISIBLE_DEVICES
    dev = "npu:0"
    rotate_quant = load_rotate_quant()
    npu_stream = torch.npu.current_stream().npu_stream

    hdr = f"{'batch':>6} {'N':>6} {'copy_us':>8} {'fht_us':>8} {'dyn_us':>8} {'rotq_us':>8}"
    print(hdr)
    print("-" * len(hdr))

    for n in DIMS:
        b = BATCH
        x = pool(lambda: torch.randn(b, n, device=dev, dtype=torch.float16))
        y = pool(lambda: torch.empty(b, n, device=dev, dtype=torch.float16))
        dy = pool(lambda: torch.empty(b, n // 2, device=dev, dtype=torch.int8))
        ds = pool(lambda: torch.empty(b, device=dev, dtype=torch.float32))

        copy_us = bench_us(lambda i: y[i % POOL].copy_(x[i % POOL]))
        fht_us = bench_us(
            lambda i: torch.ops.ascend_ops_nn.fast_hadamard(x[i % POOL], y[i % POOL])
        )
        dyn_us = bench_us(
            lambda i: torch.ops.ascend_ops_nn.fast_hadamard_dynamic_quant(
                x[i % POOL], 0, dy[i % POOL], ds[i % POOL]
            )
        )

        # rotate_quant requires K=N=n with K<=1024, N>=128.
        rotq_us = -1.0
        if rotate_quant is not None and 128 <= n <= 1024:
            rot = hadamard_matrix(n, dev)
            rq_y = pool(lambda: torch.empty(b, n, device=dev, dtype=torch.int8))
            rq_s = pool(lambda: torch.empty(b, device=dev, dtype=torch.float32))
            rotq_us = bench_us(
                lambda i: rotate_quant(
                    x[i % POOL], rot, rq_y[i % POOL], rq_s[i % POOL], npu_stream
                )
            )

        rotq = f"{rotq_us:>8.2f}" if rotq_us >= 0 else f"{'-':>8}"
        print(f"{b:>6} {n:>6} {copy_us:>8.2f} {fht_us:>8.2f} {dyn_us:>8.2f} {rotq}")


if __name__ == "__main__":
    main()
