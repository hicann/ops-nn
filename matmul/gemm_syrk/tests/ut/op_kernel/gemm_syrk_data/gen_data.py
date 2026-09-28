# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

#!/usr/bin/env python3

import argparse
import os
import numpy as np


def _fp32_to_bf16_u16(arr_fp32):
    # numpy无原生bfloat16，使用round-to-nearest-even转换后以uint16保存。
    u32 = arr_fp32.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def _bf16_u16_to_fp32(arr_bf16):
    return (arr_bf16.astype(np.uint32) << 16).view(np.float32)


def _cast_input(arr_fp32, dtype):
    if dtype == "float16":
        storage = arr_fp32.astype(np.float16)
        return storage, storage.astype(np.float32)
    storage = _fp32_to_bf16_u16(arr_fp32)
    return storage, _bf16_u16_to_fp32(storage)


def gen_gemm_syrk_data(
    m, k, batch, dtype, alpha, beta, trans, output_dir="./", seed=42
):
    """Generate gemm_syrk test data: C = alpha * (A @ A^T) + beta * C.

    The logical A is [batch, m, k]; with trans the stored input_a.bin is the
    transposed [batch, k, m] layout (kernel computes alpha * (A^T @ A)).
    C is [batch, m, m] and must be symmetric on input so that the in-place
    output stays symmetric. The golden is computed in the fp32 domain from
    the dtype-rounded inputs, then rounded back.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    a, a_fp32 = _cast_input(
        rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32), dtype
    )
    # Symmetric input C: average of a random matrix and its transpose.
    r = rng.uniform(-1.0, 1.0, (batch, m, m)).astype(np.float32)
    c_sym = 0.5 * (r + np.transpose(r, (0, 2, 1)))
    c, c_fp32 = _cast_input(c_sym, dtype)

    aat = np.matmul(a_fp32, np.transpose(a_fp32, (0, 2, 1)))
    golden_fp32 = alpha * aat + beta * c_fp32
    golden = (
        golden_fp32.astype(np.float16)
        if dtype == "float16"
        else _fp32_to_bf16_u16(golden_fp32)
    )

    a_stored = np.transpose(a, (0, 2, 1)) if trans else a
    a_stored.tofile(os.path.join(output_dir, "input_a.bin"))
    c.tofile(os.path.join(output_dir, "input_c.bin"))
    golden.tofile(os.path.join(output_dir, "golden_c.bin"))

    return a_stored, c, golden


def main():
    parser = argparse.ArgumentParser(description="Generate gemm_syrk test data")
    parser.add_argument("--m", type=int, required=True, help="M (=N) dimension")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument("--batch", type=int, default=1, help="Batch dimension")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Matmul result scale")
    parser.add_argument("--beta", type=float, default=1.0, help="C scale")
    parser.add_argument(
        "--trans",
        action="store_true",
        help="Store input_a.bin transposed as [batch, k, m]",
    )
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_gemm_syrk_data(
        args.m,
        args.k,
        args.batch,
        args.dtype,
        args.alpha,
        args.beta,
        args.trans,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
