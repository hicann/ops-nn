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
import torch


__golden__ = {
    "kernel": {"inplace_index_add_with_sorted": "inplace_index_add_with_sorted_golden"}
}


def _np_to_torch(arr):
    """numpy -> torch，兼容 bfloat16（torch.from_numpy 不直接支持 ml_dtypes.bfloat16）。"""
    if arr is None:
        return None
    if "bfloat16" in str(arr.dtype):
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(arr))


def _torch_to_np(t, ori_dtype):
    """把 torch tensor 转回 numpy，并还原原始 dtype（含 bfloat16）。"""
    if "bfloat16" in ori_dtype:
        import ml_dtypes

        return (
            t.to(torch.bfloat16)
            .view(torch.int16)
            .cpu()
            .numpy()
            .view(ml_dtypes.bfloat16)
        )
    if "float16" in ori_dtype:
        return t.to(torch.float16).cpu().numpy()
    return t.cpu().numpy()


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
        axis: int, dimension of var along which indices apply, only 0.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray) of the same shape and dtype as var.
    """
    # 统一升 fp32 计算：既规避 torch.from_numpy 不支持 bfloat16，也避免 fp16/bf16 累加精度损失
    var_t = _np_to_torch(var).to(torch.float32)
    value_t = _np_to_torch(value).to(torch.float32)
    idx_t = _np_to_torch(sorted_indices).to(torch.int64)
    pos_t = _np_to_torch(pos).to(torch.int64)

    alpha_val = 1.0
    if alpha is not None:
        alpha_val = float(_np_to_torch(alpha).item())

    # 按 pos 重排 value，再按 sorted_indices 累加到 var
    value_reordered = value_t[pos_t] * alpha_val
    out = var_t.clone()
    out.index_add_(axis, idx_t, value_reordered)

    return _torch_to_np(out, str(var.dtype))
