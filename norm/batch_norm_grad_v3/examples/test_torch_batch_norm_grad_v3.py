#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Call native_batch_norm_backward and compare with the shared CPU FP64 Golden.

Ascend 950/960: source the CANN environment and install matching torch/torch_npu.
Run from any directory:
  python3 test_torch_batch_norm_grad_v3.py --device 0
  python3 test_torch_batch_norm_grad_v3.py --dtype float16
  python3 test_torch_batch_norm_grad_v3.py --cpu

The CPU option checks native CPU Torch against Golden; it does not test the NPU.
Inputs use Torch's (N, C, H, W) convention. Saved statistics are prepared once
as inputs and supplied unchanged to both implementations. No custom Torch
extension is needed. The installed CANN/torch_npu implementation is exercised.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from assets.golden import TorchBatchNormBackwardTestSpec  # noqa: E402


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
TOLERANCES = {
    torch.float32: (1e-4, 1e-5),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (8e-3, 8e-3),
}
MASKS = (
    (True, True, True),
    (True, False, False),
    (False, True, True),
    (False, False, False),
)
EPS = 1e-5


def make_inputs(dtype):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((2, 3, 2, 2), generator=generator).to(dtype)
    dy = torch.randn(x.shape, generator=generator).to(dtype)
    weight = torch.tensor([0.5, 1.5, 2.0])
    running_mean = torch.tensor([-0.25, 0.1, 0.75])
    running_var = torch.tensor([0.25, 0.5, 1.25])
    variance, mean = torch.var_mean(x.double(), dim=(0, 2, 3), correction=0)
    return (
        dy,
        x,
        weight,
        running_mean,
        running_var,
        mean.float(),
        torch.rsqrt(variance + EPS).float(),
    )


def run_case(dtype, training, mask, device):
    inputs = make_inputs(dtype)
    expected = TorchBatchNormBackwardTestSpec.golden(*inputs, training, EPS, mask)
    actual = torch.ops.aten.native_batch_norm_backward.default(
        *(value.to(device) for value in inputs), training, EPS, list(mask)
    )
    if len(actual) != len(expected):
        raise AssertionError(f"Expected {len(expected)} outputs, got {len(actual)}")
    errors = []
    for name, got, reference in zip(("dx", "dweight", "dbias"), actual, expected):
        if reference is None:
            if got is not None:
                raise AssertionError(f"Disabled output {name} must be None")
            errors.append(f"{name}=disabled")
            continue
        if got is None:
            raise AssertionError(f"Enabled output {name} is None")
        rtol, atol = TOLERANCES[got.dtype]
        cpu_result = got.detach().cpu().double()
        torch.testing.assert_close(cpu_result, reference, rtol=rtol, atol=atol)
        errors.append(
            f"{name}: max_abs={(cpu_result - reference).abs().max().item():.3e}"
        )
    print(f"PASS dtype={dtype} train={training} mask={mask}: {', '.join(errors)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run CPU Torch only, without initializing an NPU",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="NPU device index (default: 0)"
    )
    parser.add_argument("--dtype", choices=("all", *DTYPES), default="all")
    args = parser.parse_args()
    device = torch.device("cpu")
    if not args.cpu:
        import torch_npu  # noqa: F401

        device = torch.device(f"npu:{args.device}")
        torch.npu.set_device(device)
    dtypes = DTYPES.values() if args.dtype == "all" else (DTYPES[args.dtype],)
    count = 0
    for dtype in dtypes:
        for training in (True, False):
            for mask in MASKS:
                run_case(dtype, training, mask, device)
                count += 1
    print(f"PASS: {count} cases on {device}")


if __name__ == "__main__":
    main()
