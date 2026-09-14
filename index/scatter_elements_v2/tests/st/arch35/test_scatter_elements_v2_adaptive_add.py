#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Public torch_npu correctness tests for adaptive deterministic local add.

Load the package under test before starting Python. Nonuniform small integer
updates make CPU float64 + one cast an exact oracle without masking addressing
errors behind a floating-point tolerance. This is not a performance test.
"""

import argparse
import json

import torch
import torch_npu  # noqa: F401


def cases():
    # Many independent A rows keep these tiles capacity-limited. Sorted runs
    # exercise low/high repetition and subgroup/run tails, with unchanged shape.
    for repeat in (1, 4, 8, 16, 32, 128):
        yield f"repeat_{repeat}", [1025, 2049], [1025, 2049], [1025, 2049], 1, repeat
    yield "sa_tail", [64, 1025], [2373, 1023], [2375, 1024], 0, 64
    yield "as_tail", [1025, 64], [1023, 2373], [1024, 2375], 1, 64
    yield "asa_tail", [17, 64, 65], [15, 2373, 63], [16, 2375, 64], 1, 64
    yield "small_input", [8, 3], [67, 2], [69, 4], 0, 16
    yield "short_s", [8, 1025], [31, 1023], [33, 1024], 0, 4
    # Unequal non-scatter dimensions prevent rank collapse. A sufficiently
    # large S forces capacity-limited tiles even for a small per-core A.
    for rank in range(3, 9):
        for dim in sorted({0, rank // 2, rank - 1}):
            index = [2] * rank
            first_a = next(axis for axis in range(rank) if axis != dim)
            index[first_a] = max(3, 257 >> (rank - 2))
            x = [extent + 1 for extent in index]
            src = [extent + 2 for extent in index]
            index[dim], x[dim], src[dim] = 2049, 64, 2051
            yield f"rank{rank}_dim{dim}", x, index, src, dim, 64


def check(case, dtype, index_dtype):
    name, x_shape, index_shape, src_shape, dim, repeat = case
    generator = torch.Generator().manual_seed(20260910)
    x = torch.randint(-2, 3, x_shape, generator=generator).to(dtype)
    src = torch.randint(-1, 2, src_shape, generator=generator).to(dtype)
    axis_shape = [1] * len(index_shape)
    axis_shape[dim] = index_shape[dim]
    # Use more than one input ordering, including repeated runs crossing tiles.
    axis_keys = torch.arange(index_shape[dim]) // repeat % x_shape[dim]
    order = torch.randperm(index_shape[dim], generator=generator)
    index = axis_keys[order].reshape(axis_shape).expand(index_shape).contiguous()
    expected = torch.scatter_add(x.double(), dim, index, src.double()).to(dtype)
    xn, indexn, srcn = x.npu(), index.to(index_dtype).npu(), src.npu()
    reference_bytes = None
    for _ in range(3):
        actual = torch.scatter_add(xn, dim, indexn, srcn).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_bytes = actual.contiguous().view(torch.uint8)
        if reference_bytes is not None:
            assert torch.equal(actual_bytes, reference_bytes), name
        reference_bytes = actual_bytes
    torch.testing.assert_close(xn.cpu(), x, rtol=0, atol=0)
    assert torch.equal(indexn.cpu().long(), index)
    print(
        json.dumps(
            dict(
                case=name,
                dtype=str(dtype),
                index_dtype=str(index_dtype),
                self_shape=x_shape,
                index_shape=index_shape,
                src_shape=src_shape,
                dim=dim,
                repeat=repeat,
                deterministic=True,
                status="PASS",
            )
        ),
        flush=True,
    )


def check_fp32_precision(index_dtype):
    # A large self would absorb each +1 on the serial-self-add path. Here the
    # complete S fits one tile and there is enough A to fill the local budget.
    x = torch.full((1025, 8), float(2**24), dtype=torch.float32)
    index = torch.zeros((1025, 600), dtype=torch.int64)
    src = torch.ones((1025, 600), dtype=torch.float32)
    expected = torch.scatter_add(x.double(), 1, index, src.double()).float()
    actual = torch.scatter_add(x.npu(), 1, index.to(index_dtype).npu(), src.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(
        json.dumps(
            dict(case="fp32_large_self", index_dtype=str(index_dtype), status="PASS")
        ),
        flush=True,
    )


def check_low_precision_accumulation(dtype, index_dtype):
    # A per-lane DATA_T accumulator would lose the +1 updates after the large
    # first update. The entire S fits, so there is no cross-tile cast to blame.
    x = torch.zeros((1025, 8), dtype=dtype)
    index = torch.zeros((1025, 600), dtype=torch.int64)
    src = torch.ones((1025, 600), dtype=dtype)
    src[:, 0] = 256 if dtype == torch.bfloat16 else 2048
    expected = torch.scatter_add(x.double(), 1, index, src.double()).to(dtype)
    xn, indexn, srcn = x.npu(), index.to(index_dtype).npu(), src.npu()
    first = None
    for _ in range(3):
        actual = torch.scatter_add(xn, 1, indexn, srcn).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        bits = actual.contiguous().view(torch.uint8)
        if first is not None:
            assert torch.equal(bits, first)
        first = bits
    print(
        json.dumps(
            dict(
                case="low_precision_accumulation",
                dtype=str(dtype),
                index_dtype=str(index_dtype),
                status="PASS",
            )
        ),
        flush=True,
    )


def check_empty_lane_tail(dtype, index_dtype, repeats=10):
    # Preserve the second S tile of the original failing asa_tail input. For
    # fp16/int32 its 886 indices fit one baseS tile, excluding cross-tile casts.
    # The largest key has one update, so the other subgroup lanes contribute 0.
    generator = torch.Generator().manual_seed(20260910)
    x = torch.randint(-2, 3, (17, 64, 65), generator=generator).to(dtype)
    src = torch.randint(-1, 2, (16, 2375, 64), generator=generator).to(dtype)
    keys = torch.arange(2373) // 64
    order = torch.randperm(2373, generator=generator)
    selected_keys = keys[order[886:1772]]
    assert int((selected_keys == 37).sum()) == 1
    index = selected_keys.reshape(1, 886, 1).expand(15, 886, 63).contiguous()
    src[:, :886, :] = src[:, 886:1772, :].clone()
    expected = torch.scatter_add(x.double(), 1, index, src.double()).to(dtype)
    xn, indexn, srcn = x.npu(), index.to(index_dtype).npu(), src.npu()
    first = None
    for iteration in range(repeats):
        actual = torch.scatter_add(xn, 1, indexn, srcn).cpu()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=f"empty_lane_tail iteration={iteration}",
        )
        bits = actual.contiguous().view(torch.uint8)
        if first is not None:
            assert torch.equal(bits, first), (
                "empty_lane_tail must be bitwise repeatable"
            )
        first = bits
    assert torch.equal(xn.cpu(), x)
    assert torch.equal(indexn.cpu().long(), index)
    assert torch.equal(srcn.cpu(), src)
    print(
        json.dumps(
            dict(
                case="empty_lane_tail",
                dtype=str(dtype),
                index_dtype=str(index_dtype),
                self_shape=list(x.shape),
                index_shape=list(index.shape),
                src_shape=list(src.shape),
                dim=1,
                runs=repeats,
                deterministic=True,
                status="PASS",
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--index-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--case")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.use_deterministic_algorithms(True)
    if args.case == "empty_lane_tail":
        check_empty_lane_tail(
            getattr(torch, args.dtype), getattr(torch, args.index_dtype), repeats=30
        )
        return
    selected = [case for case in cases() if args.case is None or case[0] == args.case]
    if not selected:
        parser.error("unknown case")
    for case in selected:
        check(case, getattr(torch, args.dtype), getattr(torch, args.index_dtype))
    if args.dtype == "float32" and args.case is None:
        check_fp32_precision(getattr(torch, args.index_dtype))
    elif args.case is None:
        check_low_precision_accumulation(
            getattr(torch, args.dtype), getattr(torch, args.index_dtype)
        )
    if args.case is None:
        check_empty_lane_tail(
            getattr(torch, args.dtype), getattr(torch, args.index_dtype)
        )


if __name__ == "__main__":
    main()
