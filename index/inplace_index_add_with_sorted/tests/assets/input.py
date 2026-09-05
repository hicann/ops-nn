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

import numpy as np

__input__ = {
    "kernel": {"inplace_index_add_with_sorted": "inplace_index_add_with_sorted_input"}
}


def inplace_index_add_with_sorted_input(
    var, value, sorted_indices, pos, alpha=None, *, axis=0, **kwargs
):
    """生成 sorted_indices 与 pos。

    - sorted_indices: K 个 [0, M-1] 内均匀分布、严格升序的 int32
                      （K > M 时自然产生重复，仍保持升序）
    - pos:           [0, K-1] 的随机置换（int32，无重复）

    var / value / alpha 沿用框架按 input_data_ranges 生成的数据。
    """
    M = int(var.shape[axis])
    K = int(sorted_indices.size)

    sorted_indices_new = np.linspace(0, M - 1, K, dtype=np.int32).reshape(
        sorted_indices.shape
    )
    pos_new = np.random.permutation(K).astype(np.int32).reshape(pos.shape)

    return (var, value, sorted_indices_new, pos_new, alpha)
