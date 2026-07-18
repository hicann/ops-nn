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
TTK custom golden for foreach_add_list_inplace.

Compute formula (docs/aclnnForeachAddListInplace.md:25):
    x1_i = x1_i + alpha * x2_i   (i = 0, 1, ..., n-1)

Inplace output: x1 list (n sub-tensors). Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x1_list : list of numpy arrays  (the DYNAMIC TensorList x1)
    x2_list : list of numpy arrays  (the DYNAMIC TensorList x2)
    alpha   : numpy array, shape (1,)  (scalar coefficient)
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def __golden_foreach_add_list_inplace(x1_list, x2_list, alpha, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")

    # alpha is a 1-element tensor; take the scalar value in float32 for stable compute.
    alpha_val = np.asarray(alpha).astype(np.float32).reshape(-1)[0]

    results = []
    for i, (a, b) in enumerate(zip(x1_list, x2_list)):
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)

        # Cast back to the per-output dtype declared in the CSV so the golden carries
        # the same rounding semantics as the NPU output.
        if output_dtypes is not None and i < len(output_dtypes):
            target = str(output_dtypes[i])
        else:
            target = str(np.asarray(a).dtype)

        if target == "bfloat16":
            out = a32 + alpha_val * b32
            out = out.astype(_bf16) if _bf16 is not None else out
        elif np.issubdtype(np.dtype(target), np.integer):
            # NPU int path: pure 2's-complement wraparound on BOTH alpha*x2 (mul) and
            # +x1 (add) -- NO saturation. numpy float->int cast maps overflow/NaN to
            # INT_MIN, and saturate also diverges (saturate only coincidentally matched
            # case00029_x2_with_nan). Verified 100% element match vs NPU dump on
            # case00020_alpha_extreme (alpha=INT_MIN: pure-wrap 100% vs saturate 0%)
            # and case00026_x2_extreme.
            dt = np.dtype(target)
            a_i = np.asarray(a).astype(dt)
            b_i = np.asarray(b).astype(dt)
            alpha_i = np.asarray(alpha).astype(dt).reshape(-1)[0]
            wide = np.int64 if dt.itemsize <= 4 else dt
            prod = (wide(alpha_i) * b_i.astype(wide)).astype(dt)
            out = (a_i.astype(wide) + prod.astype(wide)).astype(dt)
        else:
            out = (a32 + alpha_val * b32).astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_add_list_inplace": "__golden_foreach_add_list_inplace"}
}
