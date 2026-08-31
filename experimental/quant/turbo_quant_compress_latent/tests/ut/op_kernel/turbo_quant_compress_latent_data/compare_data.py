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

from golden import HEAD_DIM, SCALE_BYTES, slot_size


def main():
    num_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    output_mode = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    width = slot_size(HEAD_DIM, output_mode)
    packed = HEAD_DIM // 2

    if num_tokens == 0:
        # the empty-tensor case only has to prove the kernel wrote nothing and did not fault
        exp = np.fromfile("golden_slot.bin", dtype=np.uint8)
        if exp.size != 0:
            print(
                f"[FAILED] golden for an empty tensor should be empty, got {exp.size} bytes"
            )
            sys.exit(1)
        print("[SUCCESS] empty tensor produced no output")
        return

    got = np.fromfile("output_slot.bin", dtype=np.uint8)[: num_tokens * width].reshape(
        num_tokens, width
    )
    exp = np.fromfile("golden_slot.bin", dtype=np.uint8).reshape(num_tokens, width)

    # nibbles and padding are compared byte for byte
    diff = np.argwhere(got[:, :packed] != exp[:, :packed])
    pad_nonzero = int(got[:, packed + SCALE_BYTES :].sum())
    if diff.size != 0 or pad_nonzero != 0:
        print(
            f"[FAILED] mismatched nibble bytes: {diff.shape[0]}, nonzero pad bytes: {pad_nonzero}"
        )
        for row, col in diff[:16]:
            print(
                f"  token {row} byte {col}: got {got[row, col]} expected {exp[row, col]}"
            )
        sys.exit(1)

    # The stored scale (raw norm in mode 0, corrected scale in mode 1) is compared bit for bit when finite.
    # Non-finite values only have to agree on the IEEE class: a quiet NaN's payload is not fixed
    # (hardware emits 0x7FFF where numpy emits 0x7E00), and the CPU simulator's scalar fp32->fp16 cast
    # saturates non-finite values to FP16_MAX (0x7BFF) instead of propagating them, which hardware does not do.
    got_scale = (
        got[:, packed : packed + SCALE_BYTES].copy().view(np.float16).reshape(-1)
    )
    exp_scale = (
        exp[:, packed : packed + SCALE_BYTES].copy().view(np.float16).reshape(-1)
    )
    FP16_MAX = np.float16(65504.0)

    bad = []
    for i, (g, e) in enumerate(zip(got_scale, exp_scale)):
        if np.isfinite(e):
            if g.view(np.uint16) != e.view(np.uint16):
                bad.append((i, g, e, "finite scale must match bit for bit"))
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

    print(f"[SUCCESS] {num_tokens}x{width} slot bytes match the golden exactly")


if __name__ == "__main__":
    main()
