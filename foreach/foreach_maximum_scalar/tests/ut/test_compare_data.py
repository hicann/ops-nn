#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("output", "expected_status"),
    [([1.0, 2.0], 0), ([1.0, 9.0], 1), ([1.0], 1), (None, 1)],
)
def test_comparison_command_reports_missing_and_incorrect_data(
    tmp_path, output, expected_status
):
    script = (
        Path(__file__).parent / "op_kernel" / "maximum_scalar_data" / "compare_data.py"
    )
    shutil.copyfile(script, tmp_path / "compare_data.py")
    np.asarray([1.0, 2.0], dtype=np.float32).tofile(tmp_path / "golden_0.bin")
    if output is not None:
        np.asarray(output, dtype=np.float32).tofile(tmp_path / "output_0.bin")
    result = subprocess.run(
        [sys.executable, str(tmp_path / "compare_data.py"), "float32"],
        capture_output=True,
    )
    assert result.returncode == expected_status, (
        result.stdout.decode() + result.stderr.decode()
    )
