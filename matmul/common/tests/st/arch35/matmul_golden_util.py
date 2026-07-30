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
"""
Common utilities for matmul golden implementations.
"""

import numpy as np
import en_dtypes
import ml_dtypes

# ============================================================================
# dtype definitions
# ============================================================================

np_fp4_e2m1 = en_dtypes.float4_e2m1
np_fp4_e1m2 = en_dtypes.float4_e1m2
np_hif8 = en_dtypes.hifloat8
np_mx_scale = en_dtypes.float8_e8m0
np_bfloat16 = ml_dtypes.bfloat16
np_fp8_e4m3 = ml_dtypes.float8_e4m3fn
np_fp8_e5m2 = ml_dtypes.float8_e5m2
np_int4 = ml_dtypes.int4

_DTYPE_TO_STR = {
    np.float16: "float16",
    np.float32: "float32",
    np.int8: "int8",
    np.int32: "int32",
    np.uint64: "uint64",
    np_bfloat16: "bfloat16",
    np_int4: "int4",
    np_fp8_e4m3: "float8_e4m3fn",
    np_fp8_e5m2: "float8_e5m2",
    np_hif8: "hifloat8",
    np_mx_scale: "float8_e8m0",
    np_fp4_e2m1: "float4_e2m1",
    np_fp4_e1m2: "float4_e1m2",
}

_STR_TO_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
    "bfloat16": np_bfloat16,
    "hifloat8": np_hif8,
    "float8_e4m3fn": np_fp8_e4m3,
    "float8_e5m2": np_fp8_e5m2,
}


def dtype_to_str(dtype):
    """将 numpy dtype 对象转换为字符串名称。"""
    return _DTYPE_TO_STR.get(dtype, str(dtype))


def cast_output_dtype(arr, dtype_name):
    """将数组转换为目标 dtype，支持字符串或 dtype 对象。"""
    target = _STR_TO_DTYPE.get(dtype_name)
    if target is not None:
        return arr.astype(target)
    return arr.astype(dtype_name)


def nz_to_nd(data, target_shape):
    """Convert FRACTAL_NZ format to ND format.

    NZ layout: (A0, A1, ..., An, N1, M1, M0, N0)
    ND layout: (A0, A1, ..., An, M, N)

    where M = (M1-1)*M0 + pad_m, N = (N1-1)*N0 + pad_n
    """
    if len(data.shape) == 4:
        data = np.reshape(data, (1,) + data.shape)
    nd_shape = (1,) + tuple(target_shape)
    data_shape = data.shape
    m, n = nd_shape[-2:]
    N1, M1 = data_shape[-4:-2]
    M0, N0 = data_shape[-2:]
    pad_m = 1 + (m - 1) % M0
    pad_n = 1 + (n - 1) % N0
    # (A, N1, M1, M0, N0) -> (A, M1, M0, N1, N0)
    data = np.reshape(data, (np.prod(data_shape[:-4]),) + data_shape[-4:]).transpose(
        (0, 2, 3, 1, 4)
    )
    main_block = data[:, : M1 - 1, :, : N1 - 1, :]
    part_1 = data[:, M1 - 1, :pad_m, : N1 - 1, :]
    part_2 = data[:, : M1 - 1, :, N1 - 1, :pad_n]
    tail_block = data[:, M1 - 1, :pad_m, N1 - 1, :pad_n]
    # Reshape
    A = data.shape[0]
    main_block = np.reshape(main_block, (A, (M1 - 1) * M0, (N1 - 1) * N0))
    part_1 = np.reshape(part_1, (A, pad_m, (N1 - 1) * N0))
    part_2 = np.reshape(part_2, (A, (M1 - 1) * M0, pad_n))
    tail_block = np.reshape(tail_block, (A, pad_m, pad_n))
    # Concatenate
    main_concat_part1 = np.concatenate((main_block, part_1), axis=1)
    part_2_concat_tail = np.concatenate((part_2, tail_block), axis=1)
    nd = np.concatenate((main_concat_part1, part_2_concat_tail), axis=-1)
    # Reshape
    nd = np.reshape(nd, data_shape[:-4] + (m, n))
    nd = np.reshape(nd, target_shape)
    return nd


def torch_to_numpy(tensor):
    """将 PyTorch tensor 转换为 numpy array，处理 bfloat16 和 float8 等特殊类型。"""
    import torch

    if tensor is None:
        return None

    # If already a numpy array, return as-is
    if isinstance(tensor, np.ndarray):
        return tensor

    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    # Handle float8 types by converting to float32 first
    dtype_str = str(tensor.dtype)
    if "float8" in dtype_str or "Float8" in dtype_str:
        return tensor.to(torch.float32).numpy()
    return tensor.numpy()


def torch_dtype_to_str(dtype):
    """将 PyTorch dtype 转换为字符串表示（如 torch.float16 → "float16"）。"""
    import torch

    dtype_map = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.int32: "int32",
    }
    return dtype_map.get(dtype, str(dtype).replace("torch.", ""))


def hf32_truncate_np(arr):
    """Truncate a float32 numpy array to HF32 (19-bit mantissa).

    Simulates the NPU cube HF32 execution: each fp32 value's mantissa is rounded
    so that the low 12 bits of the 23-bit mantissa become 0 (round-half-up of
    the top surviving bit), keeping 11 mantissa bits.
    """
    int_view = arr.view(np.uint32)
    truncated = (((int_view >> 11) + 1) >> 1) << 12
    return truncated.view(np.float32)
