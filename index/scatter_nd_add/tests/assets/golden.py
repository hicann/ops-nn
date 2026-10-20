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


__golden__ = {"kernel": {"scatter_nd_add": "scatter_nd_add_golden"}}


def scatter_nd_add_golden(var, indices, updates, *, use_locking=False, **kwargs):
    """
    Golden function for scatter_nd_add.
    All the parameters (names and order) follow @scatter_nd_add_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    语义与 kernel 对齐 (arch35 scatter_nd_add_simt.h / arch92 scatter_nd_add_simd_dma.h):
      1. indices 最后一维为 rank, flat 行偏移 = sum(indices[m, k] * stride[k]),
         其中 stride[k] = prod(var.shape[k+1 : rank]) (host 侧 SetStride 同款定义);
      2. 越界索引 (flat < 0 或 flat >= prod(var.shape[:rank])) 静默跳过。
         TF 的 ScatterNdAdd 遇越界索引会抛 InvalidArgumentError, 无法作为含越界
         用例的 golden, 故改为 numpy 复刻 (做法同 scatter_max/scatter_div golden);
      3. 重复索引逐条累加(加法满足交换律, 顺序无关); fp16 用 float32 累加后回写;
         bool 按 kernel 的 int8 atomic-max(0/1 即逻辑或) 语义处理。

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import numpy as np

    var = np.asarray(var)
    indices = np.asarray(indices)
    updates = np.asarray(updates)

    out = var.copy()
    if var.size == 0 or indices.size == 0 or updates.size == 0:
        return out

    rank = int(indices.shape[-1])
    lead_shape = var.shape[:rank]
    after_shape = var.shape[rank:]

    var_in_axis = 1
    for d in lead_shape:
        var_in_axis *= int(d)

    # flat 行偏移, stride 定义与 host 侧 SetStride 一致
    idx2d = indices.reshape(-1, rank).astype(np.int64)
    n_idx = idx2d.shape[0]
    flat = np.zeros(n_idx, dtype=np.int64)
    for k in range(rank):
        stride = 1
        for d in var.shape[k + 1 : rank]:
            stride *= int(d)
        flat += idx2d[:, k] * stride

    valid = (flat >= 0) & (flat < var_in_axis)
    if not bool(np.any(valid)):
        return out  # 全部越界: 与 kernel 一致, var 原值输出

    upd2d = updates.reshape((n_idx,) + tuple(after_shape))
    flat_valid = flat[valid]

    if var.dtype == np.bool_:
        # kernel bool 路径为 SetAtomicMax<int8>, 0/1 取 max 即逻辑或
        acc = out.astype(np.int8).reshape((var_in_axis,) + tuple(after_shape))
        np.maximum.at(acc, flat_valid, upd2d[valid].astype(np.int8))
        return acc.reshape(var.shape).astype(np.bool_)

    if var.dtype == np.float16:
        # fp16 用 float32 累加后回写, 避免 fp16 逐条累加的中间舍入
        acc = out.astype(np.float32).reshape((var_in_axis,) + tuple(after_shape))
        np.add.at(acc, flat_valid, upd2d[valid].astype(np.float32))
        return acc.reshape(var.shape).astype(np.float16)

    # int32/int64/float32 等: 原生 dtype 累加, 溢出回绕与硬件原子加一致
    acc = out.reshape((var_in_axis,) + tuple(after_shape))
    np.add.at(acc, flat_valid, upd2d[valid])
    return acc.reshape(var.shape)
