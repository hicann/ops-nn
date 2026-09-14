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
"""Public torch_npu correctness checks for the local deterministic add path.

Run in an environment loading the package under test. This is not a latency test.
The precision oracle accumulates on CPU in float64, then converts once.
"""

import argparse
import json
import math

import torch
import torch_npu  # noqa: F401


def check_precision():
    # Each individual +1 is lost when serially added to this FP32 self value.
    # A group sum of 128 is exactly representable, including the final addition.
    x = torch.full((8, 256), float(2**24), dtype=torch.float32)
    index = torch.zeros((128, 128), dtype=torch.int64)
    src = torch.ones((129, 129), dtype=torch.float32)
    expected = torch.scatter_add(x.double(), 0, index, src.double()).float()
    actual = torch.scatter_add(x.npu(), 0, index.npu(), src.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(json.dumps({"case": "fp32_small_updates", "status": "PASS"}), flush=True)


def iter_different_shape_cases():
    # Different extents on every non-scatter axis prevent high ranks from
    # collapsing and exercise the public API's independent stride semantics.
    for rank in range(1, 9):
        for dim in range(rank):
            x_shape = [3] * rank
            index_shape = [2] * rank
            src_shape = [4] * rank
            axis_size = 32 if rank == 1 else 67
            x_shape[dim] = 8
            index_shape[dim] = axis_size
            src_shape[dim] = axis_size + 2
            yield (
                f"rank{rank}_dim{dim}",
                x_shape,
                index_shape,
                src_shape,
                dim,
                "add",
                True,
            )

    # Multi-tile S, A0 tails, and ASA tiles spanning multiple A1 positions.
    yield ("sa_tails", [64, 257], [2373, 253], [2375, 255], 0, "add", False)
    yield ("as_tails", [129, 64], [127, 2373], [128, 2375], -1, "add", False)
    yield (
        "asa_a0_tails",
        [5, 64, 257],
        [3, 2373, 253],
        [4, 2375, 255],
        1,
        "add",
        False,
    )
    yield (
        "asa_a1_tails",
        [257, 64, 3],
        [253, 2373, 2],
        [255, 2375, 4],
        -2,
        "add",
        False,
    )
    yield ("short_s_fallback", [8, 257], [31, 253], [33, 255], 0, "add", False)
    yield ("low_repeat_fallback", [64, 4096], [32, 4096], [33, 4097], 0, "add", False)
    yield ("unique_indices", [128, 257], [67, 253], [69, 255], 0, "unique", False)
    yield ("none_unchanged", [8, 257], [67, 253], [69, 255], 0, "none", False)
    yield (
        "asa_short_s_fallback",
        [3, 8, 17],
        [2, 31, 13],
        [4, 33, 15],
        1,
        "add",
        False,
    )
    yield ("none_as", [17, 8], [13, 67], [15, 69], -1, "none", False)
    yield ("none_asa", [3, 8, 17], [2, 67, 13], [4, 69, 15], 1, "none", False)


def iter_cases():
    for case in iter_different_shape_cases():
        yield case
        name, x_shape, index_shape, _, dim, mode, precision_witness = case
        # Same shape must reuse the original indices element offset, including
        # nonzero tile offsets and SA/ASA rearrangement. Keep the general-layout
        # case alongside it so an unconditional direct-address path is caught.
        yield (
            f"same_shape_{name}",
            x_shape,
            index_shape,
            list(index_shape),
            dim,
            mode,
            precision_witness,
        )
        if precision_witness:
            # The FP32 precision witness uses constant updates. Also exercise
            # every rank/axis with nonuniform values to detect wrong addresses.
            yield (
                f"same_shape_address_{name}",
                x_shape,
                index_shape,
                list(index_shape),
                dim,
                mode,
                False,
            )


def check_case(case, dtype, index_dtype):
    name, x_shape, index_shape, src_shape, dim, mode, precision_witness = case
    dim = dim % len(x_shape)
    generator = torch.Generator().manual_seed(20260908)
    x = torch.randint(-2, 3, x_shape, generator=generator).to(dtype)
    src = torch.randint(-1, 2, src_shape, generator=generator).to(dtype)
    if precision_witness and dtype == torch.float32:
        x.fill_(float(2**24))
        src.fill_(1)
    if mode == "unique":
        reshape = [1] * len(index_shape)
        reshape[dim] = index_shape[dim]
        index = (
            torch.arange(index_shape[dim])
            .reshape(reshape)
            .expand(index_shape)
            .contiguous()
        )
    else:
        # Index range is explicit and includes repeated keys with short/long groups.
        index = torch.randint(
            0, min(x_shape[dim], 32), index_shape, generator=generator
        )
    if mode == "none":
        expected = torch.scatter(x, dim, index, src)
    else:
        expected = torch.scatter_add(x.double(), dim, index, src.double()).to(dtype)
    xn, indexn, srcn = x.npu(), index.to(index_dtype).npu(), src.npu()
    if mode == "none":
        # This overload reaches aclnnScatter and accepts int32 as well as int64
        # indices; the functional scatter wrapper only accepts int64 here.
        output = torch.empty_like(xn)
        actual = torch.ops.aten.scatter.src_out(xn, dim, indexn, srcn, out=output).cpu()
        repeated = torch.ops.aten.scatter.src_out(
            xn, dim, indexn, srcn, out=output
        ).cpu()
    else:
        actual = torch.scatter_add(xn, dim, indexn, srcn).cpu()
        repeated = torch.scatter_add(xn, dim, indexn, srcn).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(
        actual.contiguous().view(torch.uint8), repeated.contiguous().view(torch.uint8)
    ), name
    torch.testing.assert_close(xn.cpu(), x, rtol=0, atol=0)
    print(
        json.dumps(
            {
                "case": name,
                "dtype": str(dtype),
                "index_dtype": str(index_dtype),
                "self_shape": x_shape,
                "index_shape": index_shape,
                "src_shape": src_shape,
                "dim": dim,
                "reduction": "none" if mode == "none" else "add",
                "elements": math.prod(index_shape),
                "status": "PASS",
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--suite", choices=("precision", "all"), default="precision")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--index-dtype", choices=("int32", "int64"), default="int64")
    parser.add_argument("--case", help="Run one named case in the full suite")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.npu.set_device(args.device)
    torch.use_deterministic_algorithms(True)
    if args.suite == "precision":
        check_precision()
    else:
        selected = [
            case for case in iter_cases() if args.case is None or case[0] == args.case
        ]
        if not selected:
            parser.error("unknown case name")
        for case in selected:
            check_case(
                case, getattr(torch, args.dtype), getattr(torch, args.index_dtype)
            )


if __name__ == "__main__":
    main()
