#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "pypto_codegen", Path(__file__).with_name("pypto_codegen.py")
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class FakeKernel:
    pass


def fake_runtime_modules():
    names = (
        "pypto_pro",
        "pypto_pro.runtime",
        "pypto_pro.runtime.opc",
        "pypto_pro.runtime.opc.pypto_compile",
        "pypto_pro.runtime.jit",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.__path__ = []
    modules["pypto_pro.runtime.jit"]._TileJitKernel = FakeKernel
    return modules


class PyptoLoaderTest(unittest.TestCase):
    def test_failed_import_restores_previous_module(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel_file = Path(directory) / "fixture_pypto_kernel.py"
            kernel_file.write_text('raise ValueError("bad kernel")\n', encoding="utf-8")
            previous = types.ModuleType("fixture_pypto_kernel")
            old_path = sys.path[:]
            old_bytecode = sys.dont_write_bytecode
            try:
                modules = fake_runtime_modules()
                modules["fixture_pypto_kernel"] = previous
                with patch.dict(sys.modules, modules):
                    with patch.object(driver, "_install_build_only_torch_stub"):
                        with self.assertRaisesRegex(ValueError, "bad kernel"):
                            driver._load_kernels(kernel_file)
                        self.assertIs(sys.modules["fixture_pypto_kernel"], previous)
            finally:
                sys.path[:] = old_path
                sys.dont_write_bytecode = old_bytecode

    def test_kernel_with_postponed_dataclass_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            kernel_file = Path(directory) / "fixture_pypto_kernel.py"
            kernel_file.write_text(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "from pypto_pro.runtime.jit import _TileJitKernel\n"
                "@dataclass\nclass Tiling:\n    rows: int\n"
                "kernel = _TileJitKernel()\n",
                encoding="utf-8",
            )
            old_path = sys.path[:]
            old_bytecode = sys.dont_write_bytecode
            try:
                with patch.dict(sys.modules, fake_runtime_modules()):
                    with patch.object(driver, "_install_build_only_torch_stub"):
                        _, kernels = driver._load_kernels(kernel_file)
                        self.assertEqual(len(kernels), 1)
                        self.assertIsInstance(kernels[0], FakeKernel)
                        self.assertEqual(
                            sys.modules["fixture_pypto_kernel"].Tiling(4).rows, 4
                        )
            finally:
                sys.path[:] = old_path
                sys.dont_write_bytecode = old_bytecode


if __name__ == "__main__":
    unittest.main()
