#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Compile and run the real SIMT header with serial CPU execution stubs.

Run with python3 tests/ut/op_kernel/arch35/test_simt_cpu.py from the operator
folder. Requires g++ (or CXX), and enables address/undefined-behavior sanitizers.
This checks the C++ algorithm; it does not replace device compilation or testing.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class SimtCpuRegression(unittest.TestCase):
    def test_index_generation(self):
        test_dir = Path(__file__).resolve().parent
        operator_dir = test_dir.parents[3]
        with tempfile.TemporaryDirectory(prefix="expand-jagged-simt-") as directory:
            build_dir = Path(directory)
            stubs = {
                "kernel_operator.h": r"""
#pragma once
#include <cstdint>
#define __aicore__
#define __simt_callee__
#define __simt_vf__
#define __gm__
#define __launch_bounds__(x)
using GM_ADDR = uint8_t*;
namespace AscendC {
struct dim3 { uint32_t x; explicit dim3(uint32_t value) : x(value) {} };
inline dim3 blockIdx(0), threadIdx(0), blockDim(1024);
template<auto Function, typename... Args> void asc_vf_call(dim3 dims, Args... args)
{
    blockDim = dims;
    for (threadIdx.x = 0; threadIdx.x < dims.x; ++threadIdx.x) {
        Function(args...);
    }
}
}
""",
                "kernel_tiling/kernel_tiling.h": "#pragma once\n",
                "simt_api/common_functions.h": "#pragma once\n",
                "ascendc/host_api/tiling/template_argument.h": (
                    "#pragma once\n#define ASCENDC_TPL_ARGS_DECL(...)\n#define ASCENDC_TPL_SEL(...)\n"
                ),
            }
            for relative, content in stubs.items():
                path = build_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            executable = build_dir / "simt_cpu_cases"
            command = shlex.split(os.environ.get("CXX", "g++")) + [
                "-std=c++17",
                "-O1",
                "-g",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-I" + str(build_dir),
                "-I" + str(operator_dir / "op_kernel/arch35"),
                str(test_dir / "simt_cpu_cases.cpp"),
                "-o",
                str(executable),
            ]
            subprocess.run(command, check=True, timeout=120)
            subprocess.run([str(executable)], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
