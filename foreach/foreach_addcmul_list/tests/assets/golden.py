#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""TTK kernel 模式自定义 golden：foreach_addcmul_list。

编写依据：docs/aclnnForeachAddcmulList.md「功能说明 / 计算公式」节：
    y_i = x1_i + scalars * x2_i * x3_i   (i = 0 .. n-1)

输入顺序（与 CSV input_shapes 一致）：x1, x2, x3, scalars
输出顺序（与 CSV output_dtypes 一致）：y

说明：shape_mapping 将每个张量列表映射为单张量（列表长度 1），scalars 为
shape [1] 的单元素张量，对应 totalTensorCount_ == 1。golden 直接对单张量计算。
"""

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:
    _bf16 = None


def __golden_foreach_addcmul_list(x1_list, x2_list, x3_list, scalars, **kwargs):
    output_dtypes = kwargs.get("output_dtypes")
    scalars_arr = np.asarray(scalars).astype(np.float32).reshape(-1)
    results = []
    for i, (a, b, c) in enumerate(zip(x1_list, x2_list, x3_list)):
        if output_dtypes is not None and i < len(output_dtypes):
            od = output_dtypes[i]
            target = od[0] if isinstance(od, (tuple, list)) else str(od)
        else:
            target = str(np.asarray(a).dtype)
        s = scalars_arr[i] if i < scalars_arr.size else scalars_arr[-1]
        if target != "bfloat16" and np.issubdtype(np.dtype(target), np.integer):
            dt = np.dtype(target)
            a_i = np.asarray(a).astype(dt)
            b_i = np.asarray(b).astype(dt)
            c_i = np.asarray(c).astype(dt)
            wide = np.int64 if dt.itemsize <= 4 else dt
            y = (
                a_i.astype(wide) + wide(s) * b_i.astype(wide) * c_i.astype(wide)
            ).astype(dt)
        else:
            # 浮点路径在 fp64 中间量算(以 fp64 真值为准,贴近 NPU FMA 的高精度中间量,
            # 避免 fp32 中间量与 NPU 的 1-ULP 假失败),再 cast 回目标 dtype。
            a64 = np.asarray(a).astype(np.float64)
            b64 = np.asarray(b).astype(np.float64)
            c64 = np.asarray(c).astype(np.float64)
            y64 = a64 + np.float64(s) * b64 * c64
            if target == "bfloat16":
                y = y64.astype(_bf16) if _bf16 is not None else y64.astype(np.float32)
            else:
                y = y64.astype(target)
        results.append(y)
    return results


__golden__ = {"kernel": {"foreach_addcmul_list": "__golden_foreach_addcmul_list"}}
