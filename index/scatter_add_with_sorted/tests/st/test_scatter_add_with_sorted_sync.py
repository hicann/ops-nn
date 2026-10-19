# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repeat FP32 scatter_add_ on 910B and compare every output with a CPU reference."""

import argparse

import torch
import torch_npu


def check_case(rows, updates, single_row, iterations):
    generator = torch.Generator().manual_seed(1234)
    initial = (torch.arange(rows * 3).reshape(rows, 3) % 11).float()
    source = (torch.arange(updates * 3).reshape(updates, 3) % 16).float()
    row_index = torch.randperm(updates, generator=generator) % rows
    if single_row:
        row_index.zero_()

    # Integer-valued FP32 inputs keep these sums exact, including the large case.
    expected = initial.clone().scatter_add_(0, row_index[:, None].expand(-1, 3), source)
    initial_npu = initial.to("npu")
    source_npu = source.to("npu")
    # Preserve stride (1, 0) so the index uses one-dimensional storage.
    index_npu = row_index.to("npu")[:, None].expand(-1, 3)
    for iteration in range(1, iterations + 1):
        actual = initial_npu.clone().scatter_add_(0, index_npu, source_npu).cpu()
        if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
            row, col = (
                (actual.view(torch.int32) != expected.view(torch.int32))
                .nonzero()[0]
                .tolist()
            )
            raise AssertionError(
                f"rows={rows}, updates={updates}, single_row={single_row}, "
                f"iteration={iteration}, position=({row}, {col}), "
                f"expected={expected[row].tolist()}, actual={actual[row].tolist()}"
            )
    print(
        f"PASS rows={rows}, updates={updates}, single_row={single_row}, calls={iterations}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument(
        "--index-refill",
        action="store_true",
        help="Also check 800001 updates once per pattern",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    torch.set_num_threads(1)
    torch_npu.npu.set_device(args.device)
    torch.use_deterministic_algorithms(True)
    for rows, updates in ((256, 512), (17, 4097)):
        for single_row in (False, True):
            check_case(rows, updates, single_row, args.iterations)
    if args.index_refill:
        for single_row in (False, True):
            check_case(17, 800001, single_row, 1)


if __name__ == "__main__":
    main()
