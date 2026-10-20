#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import torch


__golden__ = {"aclnn": {"aclnnRmsNormQuant": "aclnn_rms_norm_quant_golden"}}

_DST_SAT_MAX = {
    "int8": 127.0,
    "int4": 7.0,
    "float8_e5m2": 57344.0,
    "float8_e4m3fn": 448.0,
    "hifloat8": 32768.0,
}
_DST_SAT_MIN = {
    "int8": -128.0,
    "int4": -8.0,
    "float8_e5m2": -57344.0,
    "float8_e4m3fn": -448.0,
    "hifloat8": -32768.0,
}
_INT4_VALUES_PER_BYTE = 2
_INT4_VALUES_PER_INT32 = np.dtype(np.int32).itemsize * _INT4_VALUES_PER_BYTE


def _to_numpy(t):
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    if hasattr(t, "numpy"):
        try:
            return t.numpy()
        except (TypeError, RuntimeError):
            return t.float().numpy()
    if hasattr(t, "detach"):
        try:
            return t.detach().cpu().numpy()
        except (TypeError, RuntimeError):
            return t.detach().cpu().float().numpy()
    return np.array(t)


def _np_to_torch(arr):
    if arr is None:
        return None
    if hasattr(arr, "view") and hasattr(arr, "dim"):
        return arr
    if isinstance(arr, np.ndarray):
        dt = str(arr.dtype)
        if dt == "int4":
            # Torch has no usable CPU int4 tensor; preserve the logical NumPy dtype.
            return arr
        if np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr)
        if "float8_e5m2" in dt:
            return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e5m2)
        if "float8_e4m3" in dt:
            return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)
        if "hifloat8" in dt:
            return torch.from_numpy(arr.view(np.int8)).view(torch.int8)
        return torch.from_numpy(arr.astype(np.float32))
    return arr


def _quant_to_dtype(y_quant_np, dtype_str):
    if dtype_str not in ("int8", "hifloat8", "float8_e5m2", "float8_e4m3fn", "int4"):
        return y_quant_np

    y_quant_np = np.nan_to_num(
        y_quant_np,
        nan=0.0,
        posinf=_DST_SAT_MAX.get(dtype_str, 0.0),
        neginf=_DST_SAT_MIN.get(dtype_str, 0.0),
    )

    if dtype_str in ("int8", "int4"):
        y_quant_np = np.round(y_quant_np, 0)

    if dtype_str in _DST_SAT_MAX:
        y_quant_np = np.clip(
            y_quant_np, _DST_SAT_MIN[dtype_str], _DST_SAT_MAX[dtype_str]
        )

    if dtype_str == "int8":
        return y_quant_np.astype(np.int8)
    if dtype_str == "int4":
        from ttk.utilities.dtypes import numpy_int4

        y_quant_np = np.clip(y_quant_np, -8, 7)
        return y_quant_np.astype(numpy_int4(), copy=False)
    if dtype_str == "hifloat8":
        from en_dtypes import hifloat8

        return y_quant_np.astype(hifloat8, copy=False)
    if dtype_str == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return y_quant_np.astype(float8_e5m2, copy=False)
    if dtype_str == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return y_quant_np.astype(float8_e4m3fn, copy=False)
    return y_quant_np


def _pack_int4_to_int32(y_int4):
    """Pack the public ACLNN INT4 result into its int32 storage ABI."""
    if y_int4.ndim == 0 or y_int4.shape[-1] % _INT4_VALUES_PER_INT32 != 0:
        raise ValueError(
            f"INT4 last dimension must be divisible by {_INT4_VALUES_PER_INT32}, "
            f"but got shape {y_int4.shape}"
        )
    from ttk.utilities.dtypes import pack_4bits

    packed_bytes = pack_4bits(np.ascontiguousarray(y_int4))
    packed_shape = y_int4.shape[:-1] + (y_int4.shape[-1] // _INT4_VALUES_PER_INT32,)
    return packed_bytes.view(np.int32).reshape(packed_shape)


def aclnn_rms_norm_quant_golden(
    x, gamma, beta, scale, offset, epsilon=1e-6, y=None, dst_type=2, **kwargs
):
    x = _to_numpy(x)
    gamma = _to_numpy(gamma)
    beta = _to_numpy(beta)
    scale = _to_numpy(scale)
    offset = _to_numpy(offset)

    if scale is not None and scale.dtype != np.float32:
        scale = scale.astype(np.float32)
    if offset is not None and offset.dtype != np.float32:
        offset = offset.astype(np.float32)

    y_dtype_str = "int8"
    if y is not None:
        dt_str = str(y.dtype)
        if "int8" in dt_str:
            y_dtype_str = "int8"
        elif "hifloat8" in dt_str:
            y_dtype_str = "hifloat8"
        elif "e5m2" in dt_str:
            y_dtype_str = "float8_e5m2"
        elif "e4m3" in dt_str:
            y_dtype_str = "float8_e4m3fn"
        elif "int4" in dt_str or "int32" in dt_str:
            y_dtype_str = "int4"
    else:
        DST_TYPE_MAP = {
            2: "int8",
            29: "int4",
            34: "hifloat8",
            35: "float8_e5m2",
            36: "float8_e4m3fn",
        }
        y_dtype_str = DST_TYPE_MAP.get(dst_type, "int8")

    # An explicit INT4 output is logical INT4, while INT32 uses eight packed values.
    pack_int4 = y is None or "int32" in str(y.dtype)
    if x.size == 0:
        y_result = _quant_to_dtype(np.empty(x.shape, np.float32), y_dtype_str)
        if y_dtype_str == "int4" and pack_int4:
            y_result = _pack_int4_to_int32(y_result)
        return _np_to_torch(y_result)

    x_fp32 = x.astype(np.float32)

    # ACLNN squeezes a legal [1, H] gamma to [H] before last-axis RMSNorm.
    n = len(x.shape) - 1
    gamma_flat = gamma.reshape(-1)
    x_reshaped = x_fp32.reshape(x_fp32.shape[:n] + gamma_flat.shape)

    variance = np.mean(np.power(x_reshaped, 2), axis=n, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = 1 / std

    result_mid = x_reshaped * rstd

    gamma_fp32 = gamma.astype(np.float32) if gamma.dtype != np.float32 else gamma
    y_array = result_mid * gamma_fp32.reshape(gamma_flat.shape)

    if beta is not None:
        beta_reshaped = beta.reshape(gamma_flat.shape)
        y_array = y_array + beta_reshaped.astype(np.float32)

    y_quant = (
        y_array / scale.reshape(-1).reshape(scale.shape)
        if scale is not None
        else y_array
    )
    if offset is not None:
        y_quant = y_quant + offset.reshape(-1).reshape(offset.shape)

    y_result = _quant_to_dtype(y_quant, y_dtype_str)
    if y_dtype_str == "int4" and pack_int4:
        y_result = _pack_int4_to_int32(y_result)
    return _np_to_torch(y_result)


class AclnnRmsNormQuantTestSpec:
    """量化输出按 1 LSB 判（standard='quant'，判据实现见 comparison/quant.py）。

    y = round(x / scale) 的 round 是阶跃函数：商落在 .5 附近时，浮点末位 1 ulp 的差异
    就足以让结果跳到相邻整数格。而末位差异来自归约次序、FMA 融合、硬件舍入等实现自由度，
    不同实现之间必然存在。要求逐位相等等于要求复现某一个实现的浮点运算次序，
    故按绝对误差 <= 1 判定。
    """

    golden = staticmethod(aclnn_rms_norm_quant_golden)
    tolerance = {
        "int8": {"standard": "quant"},
        "int4": {"standard": "quant"},
        "int32": {"standard": "quant", "packed_bits": 4},
        "float8_e5m2": {"standard": "requant"},
        "float8_e4m3fn": {"standard": "requant"},
    }


__spec__ = {"aclnnRmsNormQuant": "AclnnRmsNormQuantTestSpec"}
