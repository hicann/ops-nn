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
Regression test for group_index output-initialization semantics of clipped_swiglu
torch_extension (see issue #7223 / PR #12603 review).

The kernel only writes rows covered by min(sum(group_index), batch); rows outside
the group must be deterministic zeros (golden: np.zeros + fill covered rows; st
executor: zeros_like remainder). Inputs are polluted beforehand so an
uninitialized output buffer would show garbage, making the non-zero regression
observable rather than flaky.

Covers:
  1. group case: covered rows match golden, uncovered rows are all zero;
  2. defined-but-empty group_index (sum(gi) == 0): all rows zero;
  3. group_index=None: full-write path, all rows match golden (at::empty stays
     correct when the kernel writes every row).

Run on an Ascend NPU with torch_npu and the custom whl installed, e.g.:
    python3 activation/clipped_swiglu/torch_extension/tests/test_clipped_swiglu_group_init.py
"""

import torch
import torch_npu  # noqa: F401  pylint: disable=unused-import
import cann_ops_nn_custom as cann_ops_nn

KW = dict(dim=-1, alpha=1.702, limit=7.0, bias=1.0)


def golden(x, group_rows, interleaved):
    """Reference: rows [0, group_rows) computed, the rest zeros."""
    h = x.shape[-1] // 2
    a, b = (x[..., 0::2], x[..., 1::2]) if interleaved else (x[..., :h], x[..., h:])
    A = a.clamp(max=KW["limit"])
    B = b.clamp(-KW["limit"], KW["limit"])
    y = A * torch.sigmoid(KW["alpha"] * A) * (B + KW["bias"])
    if group_rows < x.shape[0]:
        y = torch.cat([y[:group_rows], torch.zeros(x.shape[0] - group_rows, h)], dim=0)
    return y


def pollute_device_memory():
    """Run unrelated forwards so the NPU caching allocator hands out dirty blocks."""
    for _ in range(2):
        cann_ops_nn.clipped_swiglu(torch.randn(128, 512).npu(), interleaved=False, **KW)
        cann_ops_nn.clipped_swiglu(
            torch.randn(128, 512, dtype=torch.float16).npu(), interleaved=True, **KW
        )


def run_case(name, x, group_index, group_rows, interleaved):
    pollute_device_memory()
    got = cann_ops_nn.clipped_swiglu(
        x.npu(), group_index=group_index, interleaved=interleaved, **KW
    ).cpu()
    # uncovered rows must be exactly zero even on polluted memory
    if group_rows < x.shape[0]:
        assert (got[group_rows:] == 0).all(), (
            "%s: rows outside group_index leaked uninitialized memory: %s"
            % (name, got[group_rows:][got[group_rows:] != 0][:4])
        )
    # covered rows match golden (fp32 tight tolerance; kernel casts through fp32)
    exp = golden(x, group_rows, interleaved)
    diff = (
        0.0
        if group_rows == 0
        else (got[:group_rows] - exp[:group_rows]).abs().max().item()
    )
    assert diff < 1e-5, "%s: covered rows mismatch, maxdiff=%s" % (name, diff)
    print("PASS %-38s covered maxdiff=%.2e, uncovered rows all-zero" % (name, diff))


def main():
    torch.manual_seed(7)
    # 1. covered batch rows < batch: cropping rows must be zero (per-call dirty memory visible)
    for interleaved in (False, True):
        x = torch.randn(8, 64)
        x[0, :4] = 20.0  # exceed limit to verify the clamp
        gi = torch.tensor([2, 3], dtype=torch.int64).npu()  # sum=5 < batch=8
        run_case("group sum(gi)=5/8 il=%s" % interleaved, x, gi, 5, interleaved)

    # 2. valid but empty group coverage: everything zero
    x = torch.randn(4, 64)
    gi = torch.zeros(2, dtype=torch.int64).npu()  # sum=0
    run_case("group sum(gi)=0", x, gi, 0, False)

    # 3. group_index=None: non-group path, kernel writes everything
    x = torch.randn(8, 64)
    run_case("group_index=None (full write)", x, None, 8, False)

    print("All group_index initialization regression tests passed.")


if __name__ == "__main__":
    main()
