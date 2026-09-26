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

__input__ = {"kernel": {"npu_scatter_add": "npu_scatter_add_input"}}


def npu_scatter_add_input(
    x, y, s, indices, sort_idx, valid_token_num, *, use_high_precision=False, **kwargs
):
    """
    Input function for npu_scatter_add.
    All the parameters (names and order) follow @npu_scatter_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  input_ranges, full_soc_version, short_soc_version, testcase_name

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    # 1. 约束indices取值到[0, y.shape[0])内
    num_rows = y.shape[0]
    indices = np.random.randint(0, num_rows, size=indices.shape).astype(indices.dtype)

    # 2. sort_idx必须是argsort(indices)的结果（稳定排序）
    sort_idx = np.argsort(indices, kind="stable").astype(sort_idx.dtype)

    # 3. valid_token_num不超过源行数
    if valid_token_num is not None:
        valid_token_num = np.array(
            [min(max(int(valid_token_num.flat[0]), 0), x.shape[0])],
            dtype=valid_token_num.dtype,
        )

    return [x, y, s, indices, sort_idx, valid_token_num]
