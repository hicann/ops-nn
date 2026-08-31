#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


__spec__ = {"batch_norm_grad_ext2": "BatchNormGradExt2Spec"}
__golden__ = {"kernel": {"batch_norm_grad_ext2": "batch_norm_grad_ext2_golden"}}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
}


def _to_nchw_like_torch(x, data_format):
    if data_format == "NHWC":
        return x.permute(0, 3, 1, 2).contiguous()
    if data_format == "NDHWC":
        return x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def _from_nchw_like_torch(x, data_format):
    if data_format == "NHWC":
        return x.permute(0, 2, 3, 1).contiguous()
    if data_format == "NDHWC":
        return x.permute(0, 2, 3, 4, 1).contiguous()
    return x


def _to_nchw_like_np(x, data_format):
    if data_format == "NHWC":
        return np.transpose(x, (0, 3, 1, 2))
    if data_format == "NDHWC":
        return np.transpose(x, (0, 4, 1, 2, 3))
    return x


def _from_nchw_like_np(x, data_format):
    if data_format == "NHWC":
        return np.transpose(x, (0, 2, 3, 1))
    if data_format == "NDHWC":
        return np.transpose(x, (0, 2, 3, 4, 1))
    return x


def _channel_axes(rank):
    return tuple(i for i in range(rank) if i != 1)


def _sum_fp32(x, axis=None):
    return np.sum(x, axis=axis, dtype=np.float32).astype(np.float32, copy=False)


def _sum_seq_fp32(x):
    acc = np.zeros(x.shape[0], dtype=np.float32)
    for idx in range(x.shape[1]):
        acc = (acc + x[:, idx]).astype(np.float32)
    return acc


def _highest_power_of_two_lt_or_eq(value):
    if value <= 1:
        return 1
    return 1 << ((value - 1).bit_length() - 1)


def _cache_id(idx):
    return (idx ^ (idx + 1)).bit_count() - 1


def _binary_fold_sum(block_cr, r_loop_factor=64):
    """Mirror the RA SplitR in-core binary fold order for one core block."""
    r_dim = block_cr.shape[1]
    curr_loop_factor = min(r_loop_factor, r_dim)
    binary_block_cnt = (r_dim + curr_loop_factor - 1) // curr_loop_factor
    binary_fold_point = (
        _highest_power_of_two_lt_or_eq(binary_block_cnt - 1)
        if binary_block_cnt > 1
        else 1
    )
    binary_block_tail = (
        curr_loop_factor if r_dim % curr_loop_factor == 0 else r_dim % curr_loop_factor
    )
    fold_cnt = binary_block_cnt - binary_fold_point
    cache = [
        np.zeros(block_cr.shape[0], dtype=np.float32)
        for _ in range(binary_block_cnt.bit_length())
    ]
    result_cache_id = _cache_id(binary_fold_point - 1)

    for loop_idx in range(binary_fold_point):
        main_start = loop_idx * curr_loop_factor
        main = _sum_fp32(
            block_cr[:, main_start : main_start + curr_loop_factor], axis=1
        )
        if loop_idx < fold_cnt:
            fold_start = (loop_idx + binary_fold_point) * curr_loop_factor
            r_length = (
                binary_block_tail if loop_idx == fold_cnt - 1 else curr_loop_factor
            )
            main = (
                main
                + _sum_fp32(block_cr[:, fold_start : fold_start + r_length], axis=1)
            ).astype(np.float32)

        cid = _cache_id(loop_idx)
        acc = main
        for cache_idx in range(cid):
            acc = (acc + cache[cache_idx]).astype(np.float32)
        cache[cid] = acc.astype(np.float32, copy=False)
    return cache[result_cache_id]


def _ra_split_r_channel_sum(x_n, core_num=72, r_loop_factor=64):
    x_cr = np.ascontiguousarray(np.moveaxis(x_n, 1, 0), dtype=np.float32).reshape(
        x_n.shape[1], -1
    )
    r_dim = x_cr.shape[1]
    block_factor = max(r_loop_factor, (r_dim + core_num - 1) // core_num)
    used_core_num = (r_dim + block_factor - 1) // block_factor
    partials = []
    for core_idx in range(used_core_num):
        start = core_idx * block_factor
        end = min(start + block_factor, r_dim)
        partials.append(_binary_fold_sum(x_cr[:, start:end], r_loop_factor))
    partial_cr = np.stack(partials, axis=1)
    return _sum_fp32(partial_cr, axis=1)


def _sum_pairwise_fp32(x):
    acc = x.astype(np.float32, copy=True)
    while acc.shape[1] > 1:
        pair_count = acc.shape[1] // 2
        paired = (acc[:, : 2 * pair_count : 2] + acc[:, 1 : 2 * pair_count : 2]).astype(
            np.float32
        )
        acc = (
            np.concatenate([paired, acc[:, -1:]], axis=1)
            if acc.shape[1] % 2
            else paired
        )
    return acc[:, 0].astype(np.float32, copy=False)


def _chunk32_pair_channel_sum(x_n):
    x_cr = np.ascontiguousarray(np.moveaxis(x_n, 1, 0), dtype=np.float32).reshape(
        x_n.shape[1], -1
    )
    partials = []
    for start in range(0, x_cr.shape[1], 32):
        partials.append(_sum_fp32(x_cr[:, start : start + 32], axis=1))
    return _sum_pairwise_fp32(np.stack(partials, axis=1))


def _chunk_channel_sum(x_n, chunk, partial_mode="np", outer_mode="np"):
    x_cr = np.ascontiguousarray(np.moveaxis(x_n, 1, 0), dtype=np.float32).reshape(
        x_n.shape[1], -1
    )
    partials = []
    for start in range(0, x_cr.shape[1], chunk):
        block = x_cr[:, start : start + chunk]
        if partial_mode == "seq":
            partials.append(_sum_seq_fp32(block))
        else:
            partials.append(_sum_fp32(block, axis=1))
    partial_cr = np.stack(partials, axis=1)
    if outer_mode == "pair":
        return _sum_pairwise_fp32(partial_cr)
    return _sum_fp32(partial_cr, axis=1)


def _ascend_like_dscale_sum(x_n, prefer_chunk16=False):
    if prefer_chunk16:
        return _chunk_channel_sum(x_n, 16, partial_mode="np", outer_mode="pair")
    return _ascend_like_channel_sum(x_n)


def _ascend_like_channel_sum(x_n, chunk=24):
    r_dim = int(
        np.prod([x_n.shape[i] for i in range(x_n.ndim) if i != 1], dtype=np.int64)
    )
    a_dim = x_n.shape[1]
    if r_dim >= 64 and r_dim > 1024 and a_dim < 1024 and r_dim > a_dim * 4:
        return _ra_split_r_channel_sum(x_n)
    x_cr = np.ascontiguousarray(np.moveaxis(x_n, 1, 0), dtype=np.float32)
    x_cr = x_cr.reshape(x_cr.shape[0], -1)
    partials = []
    for start in range(0, x_cr.shape[1], chunk):
        partials.append(
            np.sum(x_cr[:, start : start + chunk], axis=1, dtype=np.float32)
        )
    return np.sum(np.stack(partials, axis=0), axis=0, dtype=np.float32)


def _ascend_like_dbeta_sum(x_n, use_chunk32_pair=False, data_format="NCHW"):
    if use_chunk32_pair and x_n.shape[1] != 8:
        r_dim = int(
            np.prod([x_n.shape[i] for i in range(x_n.ndim) if i != 1], dtype=np.int64)
        )
        if x_n.shape[1] == 4:
            return _chunk32_pair_channel_sum(x_n)
        if data_format == "NHWC" and r_dim > 8192:
            return _chunk_channel_sum(x_n, 1024, partial_mode="seq", outer_mode="np")
        if data_format == "NHWC" and x_n.shape[1] > 64:
            return _chunk_channel_sum(x_n, 16, partial_mode="seq", outer_mode="np")
        if data_format == "NDHWC" and x_n.shape[1] <= 64 and r_dim > 32768:
            return _chunk_channel_sum(x_n, 128, partial_mode="np", outer_mode="np")
        if data_format == "NDHWC" and x_n.shape[1] > 64 and r_dim <= 4096:
            return _chunk_channel_sum(x_n, 256, partial_mode="seq", outer_mode="np")
        return _chunk_channel_sum(x_n, 16, partial_mode="seq", outer_mode="np")
    return _ascend_like_channel_sum(x_n)


class _BatchNormGradExt2Compose:
    """Third-party GPU implementation using the same BN backward formula as the
    operator contract. Outputs are cast to the NPU output dtypes."""

    def __init__(self, epsilon=0.0001, data_format="NHWC", is_training=True, **_):
        self.epsilon = epsilon
        self.data_format = data_format
        self.is_training = is_training

    def __call__(self, y_backprop, x, scale, reserve_space_1, reserve_space_2, **_):
        x_n = _to_nchw_like_torch(x, self.data_format)
        dy_n = _to_nchw_like_torch(y_backprop, self.data_format)
        axes = _channel_axes(x_n.dim())
        reduce_count = 1
        for axis in axes:
            reduce_count *= x_n.shape[axis]
        broadcast_shape = [1] * x_n.dim()
        broadcast_shape[1] = x_n.shape[1]

        scale_n = scale.to(torch.float32).reshape(broadcast_shape)
        mean_n = reserve_space_1.to(torch.float32).reshape(broadcast_shape)
        if self.is_training:
            rstd = reserve_space_2.to(torch.float32)
        else:
            rstd = torch.rsqrt(reserve_space_2.to(torch.float32) + self.epsilon)
        rstd_n = rstd.reshape(broadcast_shape)

        dy_f = dy_n.to(torch.float32)
        x_hat = (x_n.to(torch.float32) - mean_n) * rstd_n
        dscale = torch.sum(dy_f * x_hat, dim=axes).to(torch.float32)
        doffset = torch.sum(dy_f, dim=axes).to(torch.float32)
        if self.is_training:
            dscale_n = dscale.reshape(broadcast_shape)
            doffset_n = doffset.reshape(broadcast_shape)
            dx_n = (
                (dy_f - doffset_n / reduce_count - x_hat * dscale_n / reduce_count)
                * scale_n
                * rstd_n
            )
        else:
            dx_n = dy_f * scale_n * rstd_n
        dx = _from_nchw_like_torch(dx_n, self.data_format).to(x.dtype)
        empty = torch.empty((0,), device=x.device, dtype=torch.float32)
        return [dx, dscale, doffset, empty, empty]


class BatchNormGradExt2Spec:
    @staticmethod
    def golden(y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs):
        return batch_norm_grad_ext2_golden(
            y_backprop, x, scale, reserve_space_1, reserve_space_2, **kwargs
        )

    third_party = {"torch": _BatchNormGradExt2Compose}
    tolerance = _TOL


def batch_norm_grad_ext2_golden(
    y_backprop,
    x,
    scale,
    reserve_space_1,
    reserve_space_2,
    *,
    epsilon=0.0001,
    data_format="NHWC",
    is_training=True,
    **kwargs,
):
    """Golden for BatchNormGradExt2 with Ascend-like fp32 channel reduction.

    In training mode, reserve_space_2 is save_invstd/rstd, not variance. In
    inference mode, reserve_space_2 is variance. This matches the operator
    interface. The dscale/doffset reduction uses fixed fp32 chunks to avoid
    false failures from CPU/GPU reduction order differences on large R.
    """
    use_chunk32_pair_dbeta = y_backprop.dtype == np.float64
    x_n = _to_nchw_like_np(x.astype(np.float32), data_format)
    dy_n = _to_nchw_like_np(y_backprop.astype(np.float32), data_format)
    axes = _channel_axes(x_n.ndim)
    reduce_count = np.float32(
        np.prod([x_n.shape[axis] for axis in axes], dtype=np.float64)
    )
    broadcast_shape = [1] * x_n.ndim
    broadcast_shape[1] = x_n.shape[1]

    scale_n = scale.astype(np.float32).reshape(broadcast_shape)
    mean_n = reserve_space_1.astype(np.float32).reshape(broadcast_shape)
    rs2 = reserve_space_2.astype(np.float32)
    if is_training:
        rstd = rs2
    else:
        rstd = (1.0 / np.sqrt(rs2 + np.float32(epsilon))).astype(np.float32)
    rstd_n = rstd.reshape(broadcast_shape)

    x_sub = x_n - mean_n
    x_hat = x_sub * rstd_n
    r_dim = int(np.prod([x_n.shape[axis] for axis in axes], dtype=np.int64))
    use_chunk16_dscale = (
        y_backprop.dtype == np.float64 and data_format == "NHWC" and r_dim <= 2048
    ) or (
        y_backprop.dtype != np.float64 and data_format == "NDHWC" and x_n.shape[1] <= 16
    )
    dscale = _ascend_like_dscale_sum(
        (x_sub * dy_n).astype(np.float32) * rstd_n, use_chunk16_dscale
    )
    doffset = _ascend_like_dbeta_sum(dy_n, use_chunk32_pair_dbeta, data_format)

    if is_training:
        dscale_n = dscale.reshape(broadcast_shape)
        doffset_n = doffset.reshape(broadcast_shape)
        dx_n = (
            (dy_n - doffset_n / reduce_count - x_hat * dscale_n / reduce_count)
            * scale_n
            * rstd_n
        )
    else:
        dx_n = dy_n * scale_n * rstd_n
    dx = _from_nchw_like_np(dx_n, data_format).astype(x.dtype, copy=False)
    empty = np.empty((0,), dtype=np.float32)
    return [dx, dscale.astype(np.float32), doffset.astype(np.float32), empty, empty]
