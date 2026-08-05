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

    _TORCH_FP8_TO_NP = {
        torch.float8_e4m3fn: ml_dtypes.float8_e4m3fn,
        torch.float8_e5m2: ml_dtypes.float8_e5m2,
    }
    if hasattr(torch, "hifloat8"):
        _TORCH_FP8_TO_NP[torch.hifloat8] = np_hif8
    if hasattr(torch, "float8_e8m0"):
        _TORCH_FP8_TO_NP[torch.float8_e8m0] = np_mx_scale

    np_dt = _TORCH_FP8_TO_NP.get(tensor.dtype)
    if np_dt is not None:
        return tensor.view(torch.uint8).numpy().view(np_dt)
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


# ============================================================================
# torch dtype mapping
# ============================================================================

TORCH_DTYPE_MAP = None


def _get_torch_dtype_map():
    """Lazily build torch dtype mapping to avoid import torch at module level."""
    global TORCH_DTYPE_MAP
    if TORCH_DTYPE_MAP is None:
        import torch

        TORCH_DTYPE_MAP = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8,
            "int32": torch.int32,
        }
    return TORCH_DTYPE_MAP


# ============================================================================
# torch<->numpy conversions
# ============================================================================


def numpy_to_torch(arr, dtype=None):
    """将 numpy array 转换为 PyTorch tensor，支持指定目标 dtype。"""
    import torch

    if arr is None:
        return None
    if arr.dtype == ml_dtypes.bfloat16:
        t = torch.from_numpy(arr.view(np.int16).copy()).view(torch.bfloat16)
    else:
        t = torch.from_numpy(arr.copy())
    if dtype is not None:
        t = t.to(dtype)
    return t


# ============================================================================
# Basic utilities
# ============================================================================


def transpose_last_two_dims(arr):
    """交换数组的最后两个维度（等价于矩阵转置）。"""
    return np.swapaxes(arr, -2, -1)


def get_intermediate_dtype(x_dtype_str):
    """根据输入 dtype 字符串返回对应的中间计算精度（fp16→float16, bf16→bfloat16）。"""
    if x_dtype_str == "bfloat16":
        return np_bfloat16
    elif x_dtype_str == "float16":
        return np.float16
    return None


def truncate(arr, inter_dtype):
    """将数组截断到指定中间精度（模拟硬件精度损失）。"""
    if inter_dtype is None:
        return arr
    return arr.astype(inter_dtype).astype(np.float32)


def detect_transpose_from_strides(tensor):
    """从 tensor 的 strides 推断最后两维是否转置。

    支持 torch.Tensor 和 numpy.ndarray。
    """
    if tensor is None or tensor.ndim < 2:
        return False
    import torch

    if isinstance(tensor, torch.Tensor):
        s = tensor.stride()
    else:
        itemsize = tensor.itemsize
        s = tuple(st // itemsize for st in tensor.strides)
    return s[-2] == 1 and s[-1] == tensor.shape[-2]


# ============================================================================
# NZ format utilities
# ============================================================================


def get_nz_n0(dtype_str):
    """根据 dtype 字符串返回 NZ 格式的 n0 (C0) 值。"""
    if dtype_str in ("int8", "uint8"):
        return 32
    elif dtype_str in ("float32", "int32"):
        return 2
    else:
        return 16


def nz_2d_to_nd(weight_2d, dtype_str):
    """将 NZ 格式的 2D view weight 转换为 ND 格式。"""
    m, n = weight_2d.shape
    n0 = get_nz_n0(dtype_str)
    n1 = (n + n0 - 1) // n0
    m1 = (m + 16 - 1) // 16
    weight_4d = np.ascontiguousarray(weight_2d).reshape(n1, m1, 16, n0)
    return nz_to_nd(weight_4d, (m, n))


def nd_to_nz(nd_2d, storage_shape, is_transposed):
    """将 2D ND packed 数据重排为 4D NZ 物理存储。

    匹配 C++ convert_weight_to_int4_pack 的 TransNdToNz 逻辑。

    非转置: nd(K, N_packed) → nz(n1, k1, 16, n0_packed)
      nz[n_p//n0, k//16, k%16, n_p%n0] = nd[k, n_p]

    转置:   nd(N, K_packed) → nz(k1_packed, n1, 16, n0_packed)
      nz[k_p//n0, n//16, n%16, k_p%n0] = nd[n, k_p]

    两种场景的映射公式相同:
      nz[j//n0, i//16, i%16, j%n0] = nd[i, j]
    区别在于 storage_shape 前两维的含义不同。

    Args:
        nd_2d: 2D numpy array, packed int32 数据
        storage_shape: 4D tuple, NZ 物理存储形状
        is_transposed: bool, 是否转置 (用于文档说明，实际映射公式相同)

    Returns:
        4D numpy array, NZ 物理存储数据
    """
    M, L = nd_2d.shape
    n0 = storage_shape[-1]
    nz = np.zeros(storage_shape, dtype=nd_2d.dtype)
    for i in range(M):
        for j in range(L):
            nz[j // n0, i // 16, i % 16, j % n0] = nd_2d[i, j]
    return nz


# ============================================================================
# Input generation utilities
# ============================================================================


def write_back(dst, src):
    """将 src 数据 in-place 写回 dst（torch.Tensor 用 copy_，numpy 用 copyto）。

    用于 customize_inputs 场景：框架不捕获返回值时，需要将新生成的数据
    写回原始 tensor/array，使修改对后续 NPU 执行可见。
    """
    import torch

    if isinstance(dst, torch.Tensor):
        dst.copy_(torch.from_numpy(src).to(dst.dtype))
    else:
        np.copyto(dst, src.astype(dst.dtype))
