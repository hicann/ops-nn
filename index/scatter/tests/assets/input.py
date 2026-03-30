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

__input__ = {"kernel": {"scatter": "scatter_input"}}


def scatter_input(var, indices, updates, *, axis=0, **kwargs):
    '''
    Input function for scatter.
    All the parameters (names and order) follow scatter_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: input_dtypes, full_soc_version, short_soc_version, testcase_name

    Returns:
        Input tensors
    '''
    shape, dtype = indices.shape, indices.dtype
    
    if len(indices.shape) == 1:
        indices = np.random.uniform(0, var.shape[axis] - updates.shape[axis], shape).astype(dtype)
    else:
        new_indices = []
        batch_indices_map = {}
        while True:
            batch = np.random.uniform(0, var.shape[0], (1,)).astype(dtype).item()
            if batch in batch_indices_map.keys():
                available_range = set(range(1, var.shape[axis] - updates.shape[axis]))
                for indices_start in batch_indices_map[batch]:
                    available_range -= set(
                        range(indices_start - updates.shape[axis], indices_start + updates.shape[axis]))
                if len(available_range) > 0:
                    indices_random = np.random.choice(np.array(list(available_range), dtype=object))
                    batch_indices_map[batch].append(indices_random)
                else:
                    continue
            else:
                indices_random = np.random.uniform(0, var.shape[axis] - updates.shape[axis], (1,)).astype(dtype)
                batch_indices_map[batch] = [indices_random.item()]
            new_indices.append(batch)
            if not isinstance(indices_random, int):
                indices_random = indices_random.tolist()[0]
            new_indices.append(indices_random)
            if len(new_indices) == shape[0] * 2:
                break
        indices = np.reshape(new_indices, (shape[0], 2))
    indices = indices.astype(dtype, copy=False)
    
    return [var, indices, updates]
