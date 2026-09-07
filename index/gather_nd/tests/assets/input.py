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
    "kernel": {"gather_nd": "gather_nd_input"},
    "aclnn": {"aclnnGatherNd": "aclnn_gather_nd_input"},
}


def gather_nd_input(x, indices, **kwargs):
    """Input function for gather_nd (kernel)."""
    if str(x.dtype) != "bool":
        params = np.arange(0, x.size, 1, dtype=x.dtype).reshape(x.shape)
    else:
        params = np.random.choice(a=[False, True], size=x.shape, p=[0.5, 0.5]).reshape(
            x.shape
        )

    ranks = indices.shape[-1]
    res_indices = []
    for rank in range(0, ranks):
        indices_rank = np.random.uniform(0, params.shape[rank], (1,)).astype(
            indices.dtype
        )
        res_indices.append(indices_rank.item())

    for index in indices.shape[0:-1]:
        res_indices = res_indices * index
    res_indices = np.reshape(res_indices, indices.shape).astype(
        indices.dtype, copy=False
    )

    return [params, res_indices]


def aclnn_gather_nd_input(*args, **kwargs):
    """Input function for aclnnGatherNd. Generate valid indices."""
    import torch

    params = args[0] if len(args) > 0 else kwargs.get("self")
    indices = args[1] if len(args) > 1 else kwargs.get("indices")

    if params is None or indices is None:
        return list(args)

    params_shape = params.shape
    ranks = indices.shape[-1]

    res_indices = []
    for rank in range(ranks):
        max_idx = params_shape[rank] - 1
        idx = np.random.uniform(0, max_idx + 1, (1,)).astype(np.int64)
        res_indices.append(idx.item())

    for index in indices.shape[0:-1]:
        res_indices = res_indices * index
    res_indices = np.reshape(res_indices, indices.shape).astype(np.int64)

    # Convert to torch tensor with original dtype
    indices_torch = torch.from_numpy(res_indices).to(indices.dtype)
    indices.copy_(indices_torch)
