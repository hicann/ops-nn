# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys

import numpy as np

from golden import HEAD_DIM


def main():
    num_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    packed = HEAD_DIM // 2

    if num_tokens == 0:
        # the empty-tensor case only has to prove the kernel wrote nothing and did not fault
        exp_y = np.fromfile("golden_y.bin", dtype=np.uint8)
        exp_scale = np.fromfile("golden_scale.bin", dtype=np.float16)
        if exp_y.size != 0 or exp_scale.size != 0:
            print("[FAILED] golden for an empty tensor should be empty")
            sys.exit(1)
        print("[SUCCESS] empty tensor produced no output")
        return

    got_y = np.fromfile("output_y.bin", dtype=np.uint8)[: num_tokens * packed].reshape(
        num_tokens, packed
    )
    exp_y = np.fromfile("golden_y.bin", dtype=np.uint8).reshape(num_tokens, packed)
    got_scale = np.fromfile("output_scale.bin", dtype=np.float16)
    exp_scale = np.fromfile("golden_scale.bin", dtype=np.float16)

    diff = np.argwhere(got_y != exp_y)
    if diff.size != 0:
        print(f"[FAILED] mismatched y bytes: {diff.shape[0]}")
        for row, col in diff[:16]:
            print(
                f"  token {row} byte {col}: got {got_y[row, col]} expected {exp_y[row, col]}"
            )
        sys.exit(1)

    # The scale is compared bit for bit when it is finite. Non-finite norms only have to agree on
    # the IEEE class: a quiet NaN's payload is not architecturally fixed (hardware emits 0x7FFF where
    # numpy emits 0x7E00), and the CPU simulator's scalar fp32->fp16 cast saturates non-finite values to
    # FP16_MAX (0x7BFF) instead of propagating them, which hardware does not do.
    got_norm = got_scale.reshape(-1)
    exp_norm = exp_scale.reshape(-1)
    FP16_MAX = np.float16(65504.0)

    if got_norm.size != exp_norm.size:
        print(
            f"[FAILED] scale size mismatch: got {got_norm.size}, expected {exp_norm.size}"
        )
        sys.exit(1)

    bad = []
    for i, (g, e) in enumerate(zip(got_norm, exp_norm)):
        if np.isfinite(e):
            if g.view(np.uint16) != e.view(np.uint16):
                bad.append((i, g, e, "finite norm must match bit for bit"))
        elif np.isnan(e):
            if not (np.isnan(g) or g == FP16_MAX):
                bad.append((i, g, e, "expected NaN (or the simulator's FP16_MAX)"))
        else:  # +-inf
            if not (np.isinf(g) or g == FP16_MAX):
                bad.append((i, g, e, "expected INF (or the simulator's FP16_MAX)"))
    if bad:
        print(f"[FAILED] {len(bad)} scale mismatches")
        for i, g, e, why in bad[:16]:
            print(
                f"  token {i}: got {g} (0x{g.view(np.uint16):04X}) expected {e} "
                f"(0x{e.view(np.uint16):04X}) - {why}"
            )
        sys.exit(1)

    print(
        f"[SUCCESS] {num_tokens}x{packed} y bytes and {num_tokens} scale values match the golden"
    )


if __name__ == "__main__":
    main()
