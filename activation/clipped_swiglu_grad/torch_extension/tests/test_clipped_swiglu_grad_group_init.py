# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Regression test for group_index output-initialization semantics of
clipped_swiglu_grad torch_extension (see issue #7223 / PR #12603 review).

Mirrors the forward-side test: the grad kernel only writes rows covered by
min(sum(group_index), batch); uncovered rows of grad_x must be deterministic
zeros (golden.py builds dx with np.zeros then fills covered rows). Device
memory is polluted beforehand so an uninitialized grad_x would show garbage.

Covers:
  1. group case: covered rows match golden, uncovered rows all zero;
  2. defined-but-empty group_index (sum(gi) == 0): grad_x all zeros;
  3. group_index=None: full-write path, all rows match golden.

Run on an Ascend NPU with torch_npu and the custom whl installed, e.g.:
    python3 activation/clipped_swiglu_grad/torch_extension/tests/test_clipped_swiglu_grad_group_init.py
"""

import torch
import torch_npu  # noqa: F401  pylint: disable=unused-import
import cann_ops_nn_custom as cann_ops_nn

KW = dict(dim=-1, alpha=1.702, limit=7.0, bias=1.0)


def golden_grad(grad_y, x, group_rows, interleaved):
    """Reference from tests/assets/golden.py: clamp-first + mask, zeros outside group."""
    h = x.shape[-1] // 2
    a, b = (x[..., 0::2], x[..., 1::2]) if interleaved else (x[..., :h], x[..., h:])
    A = a.clamp(max=KW["limit"])
    B = b.clamp(-KW["limit"], KW["limit"])
    s = torch.sigmoid(KW["alpha"] * A)
    da = (
        grad_y
        * (B + KW["bias"])
        * s
        * (1 + KW["alpha"] * A * (1 - s))
        * (a <= KW["limit"])
    )
    db = grad_y * A * s * ((b >= -KW["limit"]) & (b <= KW["limit"]))
    gx = (
        torch.cat([da, db], dim=-1)
        if not interleaved
        else torch.stack([da, db], dim=-1).flatten(1)
    )
    if group_rows < x.shape[0]:
        gx = torch.cat(
            [gx[:group_rows], torch.zeros(x.shape[0] - group_rows, x.shape[1])], dim=0
        )
    return gx


def pollute_device_memory():
    for _ in range(2):
        cann_ops_nn.clipped_swiglu(torch.randn(128, 512).npu(), interleaved=False, **KW)
        cann_ops_nn.clipped_swiglu_grad(
            torch.randn(128, 256, dtype=torch.float16).npu(),
            torch.randn(128, 512, dtype=torch.float16).npu(),
            interleaved=False,
            **KW,
        )


def run_case(name, grad_y, x, group_index, group_rows, interleaved):
    pollute_device_memory()
    got = cann_ops_nn.clipped_swiglu_grad(
        grad_y.npu(), x.npu(), group_index=group_index, interleaved=interleaved, **KW
    ).cpu()
    if group_rows < x.shape[0]:
        assert (got[group_rows:] == 0).all(), (
            "%s: grad_x rows outside group_index leaked uninitialized memory: %s"
            % (name, got[group_rows:][got[group_rows:] != 0][:4])
        )
    exp = golden_grad(grad_y, x, group_rows, interleaved)
    diff = (
        0.0
        if group_rows == 0
        else (got[:group_rows] - exp[:group_rows]).abs().max().item()
    )
    assert diff < 1e-4, "%s: covered rows mismatch, maxdiff=%s" % (name, diff)
    print("PASS %-38s covered maxdiff=%.2e, uncovered rows all-zero" % (name, diff))


def main():
    torch.manual_seed(7)
    # 1. sum(gi) < batch: uncovered grad_x rows must be zero, covered match golden
    for interleaved in (False, True):
        x = torch.randn(8, 64)
        x[0, :4] = 20.0
        grad_y = torch.randn(8, 32)
        gi = torch.tensor([2, 3], dtype=torch.int64).npu()
        run_case(
            "grad group sum(gi)=5/8 il=%s" % interleaved, grad_y, x, gi, 5, interleaved
        )

    # 2. empty coverage: grad_x all zeros
    x = torch.randn(4, 64)
    grad_y = torch.randn(4, 32)
    run_case(
        "grad group sum(gi)=0",
        grad_y,
        x,
        torch.zeros(2, dtype=torch.int64).npu(),
        0,
        False,
    )

    # 3. group_index=None: full write
    x = torch.randn(8, 64)
    grad_y = torch.randn(8, 32)
    run_case("grad group_index=None", grad_y, x, None, 8, False)

    print("All clipped_swiglu_grad group_index regression tests passed.")


if __name__ == "__main__":
    main()
