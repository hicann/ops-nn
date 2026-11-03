#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile a concrete-shape kernel with the real CANN compiler (no NPU needed).

After sourcing CANN and preparing build/tbe with the single-op build, run:
  python index/scatter_elements_v2/tests/st/arch22/test_static_compile.py \
      --impl-root build/tbe --output-dir /tmp/scatter-static

The output root can be reused. Each invocation retains its artifacts in a new
run-* subdirectory, so previous binaries cannot make a failed compile pass.
This exercises constexpr tiling and INFERCHANNEL, which the package's
dynamic-shape binaries do not cover. Parameter choices define smoke-test
coverage, not the operator's complete support matrix.
"""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Reusable root for per-run output directories",
    )
    parser.add_argument("--soc", default="Ascend910B2")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16", "int32"],
    )
    parser.add_argument("--indices-dtype", default="int32", choices=["int32", "int64"])
    parser.add_argument("--axis", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--reduction",
        default="none",
        choices=["none", "add", "mul", "min", "max", "mean"],
    )
    args = parser.parse_args()
    impl_root = args.impl_root.resolve(strict=True)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    print(f"Compile output directory: {output_dir}", flush=True)

    # Import after argument validation so --help works without CANN installed.
    from asc_op_compile_base.common.buildcfg import build_config
    from asc_op_compile_base.common.platform import set_current_compile_soc_info

    spec = importlib.util.spec_from_file_location(
        "scatter_static_compile", impl_root / "dynamic/scatter_elements_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    set_current_compile_soc_info(args.soc)
    os.chdir(output_dir)

    def descriptor(dtype):
        return dict(
            shape=[4, 8], ori_shape=[4, 8], format="ND", ori_format="ND", dtype=dtype
        )

    kernel_name = "scatter_static_compile"
    with build_config(
        kernel_meta_parent_dir=str(output_dir), op_debug_config="save_temp_cce_file"
    ):
        module.scatter_elements_v2(
            descriptor(args.dtype),
            descriptor(args.indices_dtype),
            descriptor(args.dtype),
            descriptor(args.dtype),
            axis=args.axis,
            reduction=args.reduction,
            kernel_name=kernel_name,
        )
    metadata = json.loads(
        (output_dir / "kernel_meta" / f"{kernel_name}.json").read_text()
    )
    binary = (
        output_dir
        / "kernel_meta"
        / (metadata["binFileName"] + metadata["binFileSuffix"])
    )
    if not binary.is_file() or binary.stat().st_size == 0:
        raise RuntimeError(
            f"Compiler did not produce a nonempty kernel binary: {binary}"
        )
    print(
        f"STATIC_COMPILE_PASS {args.soc} {args.dtype}/{args.indices_dtype} {args.reduction}: {binary}"
    )


if __name__ == "__main__":
    main()
