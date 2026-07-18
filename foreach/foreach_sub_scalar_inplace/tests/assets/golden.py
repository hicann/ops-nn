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
TTK custom golden for foreach_sub_scalar_inplace.

Compute formula (docs/aclnnForeachSubScalarInplace.md):
    x_i = x_i - scalar   (i = 0, 1, ..., n-1)

Inplace output: x list (n sub-tensors). Output order == x sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened):
    x_list : list of numpy arrays   (the DYNAMIC TensorList x)
    scalar : numpy array, shape (1,)  (scalar subtrahend)

Mirrors the proven foreach_mul_scalar_inplace golden (200/200), with - instead of *.
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def __golden_foreach_sub_scalar_inplace(x_list, scalar, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")

    # scalar is a 1-element tensor; take the scalar value in float32 for stable compute.
    scalar_val = np.asarray(scalar).astype(np.float32).reshape(-1)[0]

    results = []
    for i, a in enumerate(x_list):
        # Cast back to the per-output dtype declared in the CSV so the golden carries
        # the same rounding semantics as the NPU output.
        if output_dtypes is not None and i < len(output_dtypes):
            target = str(output_dtypes[i])
        else:
            target = str(np.asarray(a).dtype)
        if target == "bfloat16":
            a32 = np.asarray(a).astype(np.float32)
            out = a32 - scalar_val
            out = out.astype(_bf16) if _bf16 is not None else out
        elif np.issubdtype(np.dtype(target), np.integer):
            # NPU int path: pure 2's-complement wraparound on x-scalar -- NO saturation.
            # numpy float->int cast maps overflow/NaN to INT_MIN (diverges from NPU wrap),
            # so keep pure int. Same fix as foreach_mul_scalar_inplace golden.
            dt = np.dtype(target)
            a_i = np.asarray(a).astype(dt)
            scalar_i = np.asarray(scalar).astype(dt).reshape(-1)[0]
            wide = np.int64 if dt.itemsize <= 4 else dt
            out = (a_i.astype(wide) - wide(scalar_i)).astype(dt)
        else:
            a32 = np.asarray(a).astype(np.float32)
            out = (a32 - scalar_val).astype(target)
        results.append(out)
    return results


__golden__ = {
    "kernel": {"foreach_sub_scalar_inplace": "__golden_foreach_sub_scalar_inplace"}
}
