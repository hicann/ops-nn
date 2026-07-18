#!/usr/bin/env python3
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
TTK custom golden for foreach_div_list_inplace.

Compute formula (docs/aclnnForeachDivListInplace.md:25):
    x1_i = x1_i / x2_i   (i = 0, 1, ..., n-1)

Inplace output: x1 list (n sub-tensors). Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x1_list : list of numpy arrays  (the DYNAMIC TensorList x1)
    x2_list : list of numpy arrays  (the DYNAMIC TensorList x2)
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def __golden_foreach_div_list_inplace(x1_list, x2_list, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")

    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)
        # Elementwise division in float32 for stable rounding. Division by zero /
        # inf / nan follow IEEE semantics (inf / nan), matching the NPU output and
        # TTK's inf/nan digitizing for the with_inf / with_nan / zero data ranges.
        with np.errstate(divide="ignore", invalid="ignore"):
            out = a32 / b32

        # Cast back to the per-output dtype declared in the CSV so the golden carries
        # the same rounding semantics as the NPU output.
        if output_dtypes is not None and i < len(output_dtypes):
            target = str(output_dtypes[i])
        else:
            target = str(np.asarray(a).dtype)
        if target == "bfloat16":
            out = out.astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_div_list_inplace": "__golden_foreach_div_list_inplace"}
}
