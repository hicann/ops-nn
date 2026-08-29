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


__golden__ = {
    "kernel": {"inplace_index_add_with_sorted": "inplace_index_add_with_sorted_golden"}
}


def inplace_index_add_with_sorted_golden(
    var, value, sorted_indices, pos, alpha=None, *, axis=0, **kwargs
):
    """
    Golden function for inplace_index_add_with_sorted.
    All the parameters (names and order) follow @inplace_index_add_with_sorted_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Formula:
        var[sorted_indices[i], ..., :] += alpha * value[pos[i], ..., :]
    where sorted_indices are the sorted target indices along `axis`, and pos maps
    each sorted position back to the original update row of value:
        update_for_sorted_i = value[pos[i]]

    Equivalent to reordering value by pos, then index_add along axis:

        out = var.clone()
        out.index_add_(axis, sorted_indices, value[pos], alpha=alpha)

    Args:
        var: numpy.ndarray, tensor to be accumulated into (in-place semantics).
        value: numpy.ndarray, update tensor whose rows are referenced by pos.
        sorted_indices: numpy.ndarray (int32), sorted target indices along axis.
        pos: numpy.ndarray (int32), inverse permutation mapping sorted order to
            value rows.
        alpha: numpy.ndarray or None, scalar scaling factor on value. Default 1.
        axis: int, dimension of var along which indices apply.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray) of the same shape and dtype as var.
    """
    import torch

    # Keep native var dtype for accumulation; value is cast to var dtype so the
    # index_add accumulation matches the kernel's compute precision.
    tensor_var = torch.from_numpy(var.copy())
    tensor_value = torch.from_numpy(value).to(tensor_var.dtype)
    tensor_sorted_indices = torch.from_numpy(sorted_indices.ravel()).to(torch.int64)
    tensor_pos = torch.from_numpy(pos.ravel()).to(torch.int64)

    # Reorder value rows by pos: value[pos[i]] aligns with sorted_indices[i]
    tensor_updates = tensor_value[tensor_pos]

    alpha_value = 1
    if alpha is not None:
        alpha_value = torch.from_numpy(alpha).item()

    tensor_var.index_add_(
        axis, tensor_sorted_indices, tensor_updates, alpha=alpha_value
    )
    return tensor_var.numpy()
