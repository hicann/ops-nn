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

__input__ = {"kernel": {"npu_scatter_add_bwd": "npu_scatter_add_bwd_input"}}


def npu_scatter_add_bwd_input(y_grad, x, s, indices, **kwargs):
    """
    Input function for npu_scatter_add_bwd.
    All the parameters (names and order) follow @npu_scatter_add_bwd_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Returns:
        List of input tensors (length must match Input count in _def.cpp)
    """
    # 约束indices取值到[0, y_grad.shape[0])内
    num_rows = y_grad.shape[0]
    indices = np.random.randint(0, num_rows, size=indices.shape).astype(indices.dtype)

    return [y_grad, x, s, indices]
