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
TTK custom golden for foreach_addcdiv_list.

Compute formula (docs/aclnnForeachAddcdivList.md:28; regbase.h:13):
    y[t][i] = x1[t][i] + scalars[t] * (x2[t][i] / x3[t][i])   (t = 0..n-1, per list element)

Non-inplace: output y is a separate TensorList of n sub-tensors, same shape/dtype as x1.
Output order == x1 sub-tensor order.

Positional args (TTK passes context.input_arrays unflattened, in CSV input_shapes order):
    x1_list : list of numpy arrays  (DYNAMIC TensorList x1, n sub-tensors)
    x2_list : list of numpy arrays  (DYNAMIC TensorList x2, sync with x1)
    x3_list : list of numpy arrays  (DYNAMIC TensorList x3, sync with x1)
    scalars : numpy array, shape (n,)  (one scalar coefficient per list element, docs:113)
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def __golden_foreach_addcdiv_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")

    # scalars is a length-n vector: scalars[t] applies to list element t.
    scalars_arr = np.asarray(scalars).astype(np.float32).reshape(-1)

    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        a32 = np.asarray(a).astype(np.float32)
        b32 = np.asarray(b).astype(np.float32)
        c32 = np.asarray(c).astype(np.float32)
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]

        # y_i = x1_i + scalars[i] * (x2_i / x3_i)
        out = a32 + s * (b32 / c32)

        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        if target == "bfloat16":
            out = out.astype(_bf16) if _bf16 is not None else out
        else:
            out = out.astype(target)
        results.append(out)
    return results


__golden__ = {"kernel": {"foreach_addcdiv_list": "__golden_foreach_addcdiv_list"}}
