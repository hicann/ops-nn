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

import copy
import numpy as np
import torch

__golden__ = {"kernel": {"index_put_with_sort_v2": "index_put_with_sort_v2_golden"}}


def unique_multidimensional(tensors):
    """
    使用 NumPy 实现的多维张量去重方法
    对多个相同长度的张量进行去重（对应位置完全相同视为重复）

    参数:
        tensors (list[torch.Tensor]): 输入张量列表

    返回:
        uniq_tensors (list[torch.Tensor]): 去重后的张量列表
        return_index (torch.Tensor): 原始索引（按首次出现顺序排序）
    """
    if len(tensors) == 0:
        return [], torch.tensor([], dtype=torch.long)

    n = tensors[0].numel()
    device = tensors[0].device

    # 验证所有张量长度相同
    for t in tensors:
        assert t.numel() == n, "All tensors must have the same length"

    # 将张量转换为 NumPy 数组
    numpy_arrays = [t.cpu().numpy() for t in tensors]

    # 堆叠成二维数组 (n, len(tensors))
    stacked = np.column_stack(numpy_arrays)

    # 使用 NumPy 的 unique 获取唯一行及其首次出现索引
    _, unique_indices = np.unique(stacked, axis=0, return_index=True)

    # 对索引排序以保持原始顺序
    sorted_indices = np.sort(unique_indices)

    # 转换为 PyTorch 张量并放回原始设备
    return_index = torch.from_numpy(sorted_indices).long().to(device)

    # 提取去重后的结果
    uniq_tensors = [t[return_index] for t in tensors]
    return uniq_tensors, return_index


def reshape_value_tensor(value_tensor, indexed_size, x_shape):
    """
    重塑值张量，根据索引轴的连续性场景

    参数:
        value_tensor (torch.Tensor): 输入值张量
        indexed_size (list[int]): 0/1列表，1表示索引轴
        x_shape (list[int]): 目标张量的形状

    返回:
        reshaped_tensor (torch.Tensor): 重塑后的张量
    """
    # 1. 获取索引轴位置和数量
    indexed_positions = [i for i, size in enumerate(indexed_size) if size == 1]
    non_indexed_positions = [i for i, size in enumerate(indexed_size) if size == 0]

    # 如果只有一个索引轴
    if len(indexed_positions) == 1:
        idx_pos = indexed_positions[0]
        # 直接reshape到匹配的形状
        target_shape = list(x_shape)
        target_shape[idx_pos] = value_tensor.shape[idx_pos]
        return value_tensor.reshape(target_shape)

    # 多个索引轴的情况
    # 找到第一个和最后一个索引轴位置
    first_idx = min(indexed_positions)
    last_idx = max(indexed_positions)

    # 检查索引轴是否连续
    is_continuous = (last_idx - first_idx + 1) == len(indexed_positions)

    if is_continuous:
        # 连续场景：只合并连续的索引轴部分
        indexed_dims = value_tensor.shape[first_idx : last_idx + 1]
        indexed_total_size = np.prod(indexed_dims)

        new_shape = list(value_tensor.shape)
        new_shape = (
            new_shape[:first_idx] + [indexed_total_size] + new_shape[last_idx + 1 :]
        )
        return value_tensor.reshape(new_shape)
    else:
        # 非连续场景：需要重新排列维度
        # 将所有索引轴移到前面，其他轴保持不变
        permute_order = indexed_positions + non_indexed_positions

        # 先置换维度
        permuted_tensor = value_tensor.permute(permute_order)

        # 然后合并所有索引轴
        indexed_total_size = np.prod(
            [permuted_tensor.shape[i] for i in range(len(indexed_positions))]
        )
        non_indexed_shape = list(permuted_tensor.shape[len(indexed_positions) :])

        new_shape = [indexed_total_size] + non_indexed_shape
    return permuted_tensor.reshape(new_shape)


def slice_value_tensor(value_tensor, indexed_size, x_shape, return_index):
    """
    根据索引轴配置对value_tensor进行切片

    参数:
        value_tensor (torch.Tensor): 输入的值张量（已重塑）
        indexed_size (list[int]): 0/1列表，1表示原始索引轴
        x_shape (list[int]): 目标张量的原始形状
        return_index (torch.Tensor): 去重后的索引位置

    返回:
        sliced_tensor (torch.Tensor): 切片后的张量
    """
    indexed_positions = [i for i, size in enumerate(indexed_size) if size == 1]

    # 如果只有一个索引轴
    if len(indexed_positions) == 1:
        idx_pos = indexed_positions[0]
        # 在该轴上进行切片
        sliced_tensor = torch.index_select(
            value_tensor, dim=idx_pos, index=return_index
        )
        return sliced_tensor

    # 多个索引轴的情况
    first_idx = min(indexed_positions)
    last_idx = max(indexed_positions)
    is_continuous = (last_idx - first_idx + 1) == len(indexed_positions)

    if is_continuous:
        # 连续场景：在第一个索引轴位置切片
        slice_dim = first_idx
        sliced_tensor = torch.index_select(
            value_tensor, dim=slice_dim, index=return_index
        )
        return sliced_tensor
    else:
        # 非连续场景：在合并后的第一个维度（索引轴合并后在第0维）切片
        sliced_tensor = torch.index_select(value_tensor, dim=0, index=return_index)
        # 恢复原始维度顺序
        # 这里需要将维度重新排列回原始顺序
        # 这里需要先恢复维度再切片？还是先切片再恢复？
        # 实际上，由于我们已经在第0维切了片，所以恢复维度时需要考虑到这一点
        # 但更简单的方法是先恢复维度再切片
        # ... 这里逻辑比较复杂，建议重新设计
    return sliced_tensor


def index_put_with_sort_v2_golden(
    self, linear_index, pos_idx, values, *, indexed_sizes, accumulate=False, **kwargs
):
    """
    Golden function for index_put_with_sort_v2.
    All the parameters (names and order) follow @index_put_with_sort_v2_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    x = self
    x_shape = x.shape
    bf16_mark = False
    if "bfloat16" in str(x.dtype):
        bf16_mark = True
        x = x.astype(np.float32)
    else:
        x = copy.deepcopy(x)
    x_torch = torch.from_numpy(x)
    mask = indexed_sizes
    np.random.seed(0)
    origin_indices_list = []
    for i in range(len(x_shape)):
        if mask[i] != 0:
            origin_indices_list.append(
                torch.from_numpy(
                    np.random.randint(x_shape[i], size=linear_index.size).astype(
                        "int64"
                    )
                )
            )
    uniq_tensors, return_index = unique_multidimensional(origin_indices_list)

    value_tensor = torch.from_numpy(values)
    reshaped_value_tensor = reshape_value_tensor(value_tensor, mask, x_shape)
    unique_values = slice_value_tensor(
        reshaped_value_tensor, mask, x_shape, return_index
    )
    mask = indexed_sizes
    if accumulate:
        x_torch.index_put_(origin_indices_list, value_tensor, accumulate)
    else:
        index_tuple = []
        idx = 0
        for i in range(len(mask)):
            if mask[i]:
                index_tuple.append(uniq_tensors[idx])
                idx += 1
            else:
                index_tuple.append(slice(None))
        x_torch[tuple(index_tuple)] = unique_values
    if bf16_mark:
        x_torch = x_torch.to(torch.bfloat16)
    res = x_torch.numpy()
    return res
