#!/usr/bin/env python3
# coding: utf-8
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
PyPTO kernel codegen driver.

This script is invoked from CMake configure stage for operators marked with enable_pypto_kernel(<op_file>).
It imports the kernel python file, runs host-side binary header generation, and copies generated infer
artifacts into --out-dir.
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock


def _install_build_only_torch_stub():
    """Stub runtime-only dependencies that binary codegen does not use.

    Lazy and idempotent: only inject when the real module is absent, and never
    overwrite an existing sys.modules entry.
    """
    if importlib.util.find_spec("torch") is None:
        sys.modules.setdefault("torch", MagicMock(name="torch"))
    if importlib.util.find_spec("torch_npu") is None:
        sys.modules.setdefault("torch_npu", MagicMock(name="torch_npu"))


def _disable_bytecode_writes():
    """Keep importing the kernel from dropping __pycache__ into the operator's op_kernel dir.

    kernel_src_copy() copies that directory verbatim into the package, so a stray .pyc would ship.
    """
    sys.dont_write_bytecode = True


# soc (ASCEND_COMPUTE_UNIT, lowercased) -> pypto arch token used by PYPTO_JIT_ARCH.
# Sole authoritative source: cmake/func.cmake's pypto_soc_to_arch() consumes this via --print-arch-map,
# so the two sides cannot drift again. Mirrors build.sh's SOC_TO_ARCH: 2201 -> a3 (ascend910b/ascend910_93),
# 3510 -> a5 (ascend950/ascend350).
_ARCH_MAP = {
    "ascend910b": "a3",
    "ascend910_93": "a3",
    "ascend950": "a5",
    "ascend350": "a5",
}


def _load_kernels(py_file: Path):
    """Import the kernel module and return all @pl.jit kernel objects defined in it."""
    sys.path.insert(0, str(py_file.parent))
    _disable_bytecode_writes()
    _install_build_only_torch_stub()

    import pypto_pro.runtime.opc.pypto_compile as pto_compile
    from pypto_pro.runtime.jit import _TileJitKernel

    spec = importlib.util.spec_from_file_location(py_file.stem, str(py_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kernels = [v for v in vars(module).values() if isinstance(v, _TileJitKernel)]
    return pto_compile, kernels


def main():
    parser = argparse.ArgumentParser(description="PyPTO kernel codegen driver")
    parser.add_argument("--py-file", required=False, help="kernel python source file")
    parser.add_argument(
        "--out-dir", required=False, help="directory to place generated artifacts"
    )
    parser.add_argument(
        "--op-file",
        required=False,
        help="op file stem (e.g. flash_attention_score_apt)",
    )
    parser.add_argument(
        "--soc", required=False, help="ascend compute unit (e.g. ascend950)"
    )
    parser.add_argument(
        "--print-arch-map",
        action="store_true",
        help="print the soc -> arch mapping (one 'soc=arch' per line) and exit",
    )
    args = parser.parse_args()

    if args.print_arch_map:
        for soc, arch in sorted(_ARCH_MAP.items()):
            print(f"{soc}={arch}")
        return

    if not (args.py_file and args.out_dir and args.op_file and args.soc):
        parser.error(
            "--py-file, --out-dir, --op-file and --soc are required unless --print-arch-map is used"
        )

    soc = args.soc.strip().lower()
    arch = _ARCH_MAP.get(soc)
    if arch is None:
        # Failing loudly beats shipping a kernel whose arch pypto auto-detected differently.
        raise SystemExit(
            f"[pypto_codegen] no PYPTO_JIT_ARCH mapping for soc '{soc}'; "
            f"supported: {sorted(_ARCH_MAP)}"
        )
    os.environ["PYPTO_JIT_ARCH"] = arch  # pypto reads this during compile

    py_file = Path(args.py_file).resolve()
    if not py_file.is_file():
        raise SystemExit(f"[pypto_codegen] py-file not found: {py_file}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pto_compile, kernels = _load_kernels(py_file)
    if not kernels:
        raise SystemExit(f"[pypto_codegen] no @pl.jit kernel found in {py_file}")

    copied = []
    for kernel in kernels:
        # generate_binary_headers() emits host tiling headers plus a self-contained cpp used only for .i infer.
        # The build-stage opc still produces the real per-tilingkey kernel sources.
        binary_dir = Path(pto_compile.generate_binary_headers(kernel)).resolve()
        for pattern in ("*_tiling.h", "*_tilingkey.h", "*_pypto_infer.cpp"):
            for src in sorted(binary_dir.glob(pattern)):
                shutil.copy(src, out_dir / src.name)
                copied.append(src.name)

    logging.info("%s: generated %s into %s", args.op_file, sorted(set(copied)), out_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    main()
