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


__golden__ = {
    "kernel": {"rms_norm_quant_v3": "rms_norm_quant_v3"},
    "aclnn": {"aclnnRmsNormQuantV3": "aclnn_rms_norm_quant_v3_golden"},
}
__spec__ = {
    "rms_norm_quant_v3": "RmsNormQuantV3TestSpec",
    "aclnnRmsNormQuantV3": "AclnnRmsNormQuantV3TestSpec",
}


def rms_norm_quant_v3(
    input_x,
    input_gamma,
    input_scales1,
    input_scales2=None,
    input_zero_points1=None,
    input_zero_points2=None,
    input_beta=None,
    epsilon=1e-6,
    div_mode=True,
    output_dtypes=None,
    dst_type=2,
    output_rstd=False,
    **kwargs,
):
    DST_TYPE_MAP = {
        2: "int8",
        29: "int4",
        34: "hifloat8",
        35: "float8_e5m2",
        36: "float8_e4m3fn",
    }
    if output_dtypes is None:
        quant_dtype = DST_TYPE_MAP.get(dst_type, "int8")
        output_dtypes = (quant_dtype, quant_dtype, "float32")
    y1_dtype = str(output_dtypes[0]) if output_dtypes is not None else "int8"
    y2_dtype = (
        str(output_dtypes[1])
        if (output_dtypes is not None and len(output_dtypes) > 1)
        else None
    )

    # Flatten only the normalization axes. Quantization parameters describe
    # their own suffix of x, which need not have the same rank as gamma.
    original_x_shape = input_x.shape
    norm_dim_num = max(1, input_gamma.ndim)
    n = input_x.ndim - norm_dim_num
    input_gamma = input_gamma.reshape(-1)
    x_shape = input_x.shape[0:n] + input_gamma.shape
    input_x = input_x.reshape(x_shape)

    len_shape_x = len(input_x.shape)
    len_shape_gamma = len(input_gamma.shape)
    axis = len_shape_x - len_shape_gamma
    input_x_dtype = input_x.dtype
    input_scales_dtype = input_scales1.dtype
    input_zp_dtype = None
    if input_zero_points1 is not None:
        input_zp_dtype = input_zero_points1.dtype
    if input_zero_points2 is not None:
        input_zp_dtype = input_zero_points2.dtype

    if input_scales_dtype is not np.float32:
        input_scales1 = input_scales1.astype(np.float32)
    if input_scales2 is not None:
        if input_scales_dtype is not np.float32:
            input_scales2 = input_scales2.astype(np.float32)
    if input_zero_points1 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points1 = input_zero_points1.astype(np.float32)
    if input_zero_points2 is not None:
        if input_zp_dtype == np.int32:
            input_zero_points2 = input_zero_points2.astype(np.float32)
        elif input_scales_dtype is not np.float32:
            input_zero_points2 = input_zero_points2.astype(np.float32)

    x_fp32 = input_x.astype(np.float32)
    variance = np.mean(np.power(x_fp32, 2), axis=axis, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd = 1 / std
    result_mid = x_fp32 * rstd

    if input_x_dtype == np.float32:
        y_array = result_mid * input_gamma
    elif input_x_dtype == np.float16:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32
    else:
        input_gamma_fp32 = input_gamma.astype(np.float32)
        y_array = result_mid * input_gamma_fp32

    if input_beta is not None:
        beta_reshaped = input_beta.reshape(input_gamma.shape)
        y_array = y_array + beta_reshaped.astype(np.float32)

    # Restore x's axes before broadcasting scalar or multidimensional scales
    # and zero points; never index shape[0] on a scalar quantization parameter.
    y_array = y_array.reshape(original_x_shape)

    if div_mode:
        input_scales1 = 1.0 / input_scales1
        if input_scales2 is not None:
            input_scales2 = 1.0 / input_scales2

    y1_quant = y_array * input_scales1
    y2_quant = None
    if input_zero_points1 is not None:
        y1_quant = y1_quant + input_zero_points1

    if input_scales2 is not None:
        y2_quant = y_array * input_scales2
    if input_zero_points2 is not None:
        y2_quant = y2_quant + input_zero_points2

    def quant_process(y_quant, q_dtype):
        y_quant = np.nan_to_num(
            y_quant,
            nan=0.0,
            posinf=_DST_SAT_MAX.get(q_dtype, 0.0),
            neginf=_DST_SAT_MIN.get(q_dtype, 0.0),
        )
        if q_dtype in _DST_SAT_MAX:
            y_quant = np.clip(y_quant, _DST_SAT_MIN[q_dtype], _DST_SAT_MAX[q_dtype])
        if q_dtype in ("int8", "int4"):
            y_quant = y_quant.astype(np.float16).astype(np.float32)
            y_quant = np.round(y_quant, 0)
        elif q_dtype not in ("hifloat8", "float8_e5m2", "float8_e4m3fn"):
            y_quant = np.round(y_quant, 0)
        if q_dtype == "int8":
            y_quant = y_quant.clip(-128, 127)
            y_quant = y_quant.astype("int8", copy=False)
        elif q_dtype == "int4":
            from ttk.utilities.dtypes import numpy_int4

            y_quant = np.clip(y_quant, -8, 7)
            y_quant = y_quant.astype(numpy_int4(), copy=False)
        elif q_dtype == "int32":
            y_quant = np.nan_to_num(y_quant, nan=0)
            y_quant = np.clip(y_quant, -2147483648, 2147483647)
            y_quant = y_quant.astype("int32", copy=False)
        elif q_dtype == "float8_e5m2":
            from ml_dtypes import float8_e5m2 as numpy_float8_e5m2

            y_quant = y_quant.astype(numpy_float8_e5m2, copy=False)
        elif q_dtype == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn as numpy_float8_e4m3fn

            y_quant = y_quant.astype(numpy_float8_e4m3fn, copy=False)
        elif q_dtype == "hifloat8":
            from en_dtypes import hifloat8 as numpy_hifloat8

            y_quant = y_quant.astype(numpy_hifloat8, copy=False)
        return y_quant

    rstd_shape = original_x_shape[:n] + (1,) * norm_dim_num
    rstd_out = (
        rstd.reshape(rstd_shape).astype(np.float32, copy=False) if output_rstd else None
    )
    return (
        quant_process(y1_quant, y1_dtype),
        quant_process(y2_quant, y2_dtype) if y2_quant is not None else None,
        rstd_out,
    )


class RmsNormQuantV3TestSpec:
    """整数量化输出按 1 LSB 判；FP8 输出按相邻量化码判。

    y = round(x / scale) 的 round 是阶跃函数：商落在 .5 附近时，浮点末位 1 ulp 的差异
    就足以让结果跳到相邻整数格。而末位差异来自归约次序、FMA 融合、硬件舍入等实现自由度，
    不同实现之间必然存在。要求逐位相等等于要求复现某一个实现的浮点运算次序，
    故整数按绝对误差 <= 1 判定，FP8 按 requant 的码距判定。
    """

    golden = staticmethod(rms_norm_quant_v3)
    tolerance = {
        "int8": {"standard": "quant"},
        "int4": {"standard": "quant"},
        "float8_e5m2": {"standard": "requant"},
        "float8_e4m3fn": {"standard": "requant"},
        "float32": {"standard": "stat_rel_err"},
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


# 形参表必须逐位对齐 aclnnRmsNormQuantV3 的原型：属性（epsilon/divMode/outputRstd）
# 排在输出 tensor（y/rstd）**前面**。把 y/rstd 提前会让 epsilon 收到一个 tensor，
# 典型报错 'float' object has no attribute 'dtype' / 精度整片错。
# dst_type 不在原型里，只能挂末尾走 extra_attrs。
def aclnn_rms_norm_quant_v3_golden(
    x,
    gamma,
    scale,
    offset,
    beta,
    epsilon=1e-6,
    div_mode=True,
    output_rstd=False,
    y=None,
    rstd=None,
    dst_type=2,
    **kwargs,
):
    divMode = div_mode
    outputRstd = output_rstd
    x = _to_numpy(x)
    gamma = _to_numpy(gamma)
    scale = _to_numpy(scale)
    offset = _to_numpy(offset)
    beta = _to_numpy(beta)

    if scale is not None and scale.dtype != np.float32:
        scale = scale.astype(np.float32)
    if offset is not None and offset.dtype != np.float32:
        offset = offset.astype(np.float32)

    x_fp32 = x.astype(np.float32)

    # ACLNN squeezes a legal [1, H] gamma to [H] before last-axis RMSNorm.
    n = len(x.shape) - 1
    gamma_flat = gamma.reshape(-1)
    x_reshaped = x_fp32.reshape(x_fp32.shape[:n] + gamma_flat.shape)

    variance = np.mean(np.power(x_reshaped, 2), axis=n, keepdims=True)
    std = np.sqrt(variance + epsilon)
    rstd_val = 1 / std

    result_mid = x_reshaped * rstd_val

    gamma_fp32 = gamma.astype(np.float32) if gamma.dtype != np.float32 else gamma
    y_array = result_mid * gamma_fp32.reshape(gamma_flat.shape)

    if beta is not None:
        beta_reshaped = beta.reshape(gamma_flat.shape)
        y_array = y_array + beta_reshaped.astype(np.float32)

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

    if divMode:
        y_quant = (
            y_array / scale.reshape(-1).reshape(scale.shape)
            if scale is not None
            else y_array
        )
    else:
        y_quant = (
            y_array * scale.reshape(-1).reshape(scale.shape)
            if scale is not None
            else y_array
        )
    if offset is not None:
        y_quant = y_quant + offset.reshape(-1).reshape(offset.shape)

    y_result = _quant_to_dtype(y_quant, y_dtype_str)
    if y_dtype_str == "int4":
        y_result = _pack_int4_to_int32(y_result)
    y_t = _np_to_torch(y_result)

    if outputRstd:
        rstd_np = rstd_val.astype(np.float32)
        rstd_t = _np_to_torch(rstd_np)
        return [y_t, rstd_t]

    return [y_t, None]


class AclnnRmsNormQuantV3TestSpec:
    """量化输出按 1 LSB 判（standard='quant'，判据实现见 comparison/quant.py）。

    y = round(x / scale) 的 round 是阶跃函数：商落在 .5 附近时，浮点末位 1 ulp 的差异
    就足以让结果跳到相邻整数格。而末位差异来自归约次序、FMA 融合、硬件舍入等实现自由度，
    不同实现之间必然存在。要求逐位相等等于要求复现某一个实现的浮点运算次序，
    故按绝对误差 <= 1 判定。
    """

    golden = staticmethod(aclnn_rms_norm_quant_v3_golden)
    tolerance = {
        "int8": {"standard": "quant"},
        "int4": {"standard": "quant"},
        "int32": {"standard": "quant", "packed_bits": 4},
        "float8_e5m2": {"standard": "requant"},
        "float8_e4m3fn": {"standard": "requant"},
        # rstd is already produced as float32 by the golden.  Avoid whole-call
        # dtype promotion, which cannot cross the Torch boundary for a
        # HIFLOAT8 sibling output.
        "float32": {"standard": "stat_rel_err"},
    }
