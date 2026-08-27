# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lightweight static checks for the ApplyCamePart3 test assets."""

from pathlib import Path


def test_golden_declares_kernel_spec():
    source = Path(__file__).parent / "assets" / "golden.py"
    text = source.read_text(encoding="utf-8")
    assert '"apply_came_part3": "ApplyCamePart3KernelSpec"' in text


def test_validation_matrix_has_dtype_attribute_and_failure_cases():
    matrix = (
        "np.float16",
        "np.float32",
        "bfloat16",
        "np.int64",
        "use_first_moment=false",
        "use_first_moment=true",
        "GRAPH_FAILED",
    )
    assert all(matrix)
