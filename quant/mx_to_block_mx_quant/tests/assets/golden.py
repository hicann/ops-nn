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

import numpy

__golden__ = {"kernel": {"mx_to_block_mx_quant": "mx_to_block_mx_quant_golden"}}

DATA_TYPE_INT_TO_STR = {
    0: "float32",
    1: "float16",
    2: "int8",
    3: "int32",
    4: "uint8",
    6: "int16",
    7: "uint16",
    8: "uint32",
    9: "int64",
    10: "uint64",
    11: "double",
    12: "bool",
    16: "complex64",
    17: "complex128",
    18: "qint8",
    19: "qint16",
    20: "qint32",
    21: "quint8",
    22: "quint16",
    23: "resource",
    25: "dual",
    26: "variant",
    27: "bfloat16",
    29: "int4",
    30: "uint1",
    31: "int2",
    32: "uint2",
    33: "complex32",
    34: "hifloat8",
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    37: "float8_e8m0",
    40: "float4_e2m1",
    41: "float4_e1m2",
    42: "hifloat4",
}

E8M0_EXP_BIAS = 127
E8M0_EXP_MIN = -127
E8M0_EXP_MAX = 127


def _numpy_mx_dtype(mx_ele_dtype):
    if mx_ele_dtype == "float4_e2m1":
        from en_dtypes import float4_e2m1

        return float4_e2m1
    if mx_ele_dtype == "float4_e1m2":
        from en_dtypes import float4_e1m2

        return float4_e1m2
    if mx_ele_dtype == "float8_e4m3fn":
        from ml_dtypes import float8_e4m3fn

        return float8_e4m3fn
    if mx_ele_dtype == "float8_e5m2":
        from ml_dtypes import float8_e5m2

        return float8_e5m2
    if mx_ele_dtype == "float8_e8m0":
        from en_dtypes import float8_e8m0

        return float8_e8m0
    raise ValueError(f"Unsupported mx dtype: {mx_ele_dtype}")


def _get_max_offset(x_dtype: str, y_dtype: str) -> int:
    if x_dtype == "float4_e2m1":
        return 13 if y_dtype == "float8_e5m2" else 6
    if x_dtype == "float4_e1m2":
        return 15 if y_dtype == "float8_e5m2" else 8
    raise ValueError(f"Unsupported x dtype for MAX_OFFSET lookup: {x_dtype}")


def _decode_e8m0_to_exp(tensor: numpy.ndarray) -> numpy.ndarray:
    """Decode FP8_E8M0 tensor to exponent values (log2)."""
    raw = tensor.view(numpy.uint8)
    exp = raw.astype(numpy.float32) - E8M0_EXP_BIAS
    return exp


def _encode_exp_to_e8m0(exp: numpy.ndarray) -> numpy.ndarray:
    """Encode exponent values back to FP8_E8M0 tensor."""
    exp = numpy.clip(exp, E8M0_EXP_MIN, E8M0_EXP_MAX)
    raw = (exp + E8M0_EXP_BIAS).astype(numpy.uint8)
    return raw.view(_numpy_mx_dtype("float8_e8m0"))


def _unpack_col_blocks(tensor: numpy.ndarray, col_blocks: int) -> numpy.ndarray:
    """Unpack last two dims [ceil(col_blocks/2), 2] -> [col_blocks]."""
    leading = list(tensor.shape[:-2])
    flat = tensor.reshape(leading + [-1])
    return flat[..., :col_blocks]


def _pack_scale1(scale1_exp: numpy.ndarray, col_blocks: int) -> numpy.ndarray:
    """Pack scale1 from [..., rows, col_blocks] to [..., rows, ceil(col_blocks/2), 2]."""
    if col_blocks % 2 != 0:
        pad_width = [(0, 0)] * (scale1_exp.ndim - 1) + [(0, 1)]
        scale1_exp = numpy.pad(
            scale1_exp, pad_width, mode="constant", constant_values=E8M0_EXP_MIN
        )
    new_shape = list(scale1_exp.shape[:-1]) + [scale1_exp.shape[-1] // 2, 2]
    return scale1_exp.reshape(new_shape)


def _pack_scale2(scale2_exp: numpy.ndarray, row_blocks: int) -> numpy.ndarray:
    """Pack scale2 from [..., row_blocks, cols] to [..., ceil(row_blocks/2), cols, 2] with interleaving."""
    if row_blocks % 2 != 0:
        pad_width = [(0, 0)] * (scale2_exp.ndim - 2) + [(0, 1), (0, 0)]
        scale2_exp = numpy.pad(
            scale2_exp, pad_width, mode="constant", constant_values=E8M0_EXP_MIN
        )
        row_blocks_padded = row_blocks + 1
    else:
        row_blocks_padded = row_blocks

    leading = list(scale2_exp.shape[:-2])
    cols = scale2_exp.shape[-1]
    reshaped = scale2_exp.reshape(leading + [row_blocks_padded // 2, 2, cols])

    transpose_order = list(range(reshaped.ndim - 3)) + [
        reshaped.ndim - 3,
        reshaped.ndim - 1,
        reshaped.ndim - 2,
    ]
    interleaved = reshaped.transpose(transpose_order)
    return interleaved


def _mx_to_block_mx_quant(x, mxscale, *, y_dtype_str="float8_e5m2"):
    """
    Core computation: convert mx format to block mx format.

    Args:
        x: input tensor (float4_e2m1 or float4_e1m2), numpy array 2D or 3D
        mxscale: scale tensor (float8_e8m0), numpy array
        y_dtype_str: output dtype string, "float8_e5m2" or "float8_e4m3fn"

    Returns:
        y: quantized output tensor
        scale1: packed scale1 tensor
        scale2: packed scale2 tensor
    """
    x_dtype = x.dtype.name

    if not isinstance(x, numpy.ndarray):
        raise RuntimeError(
            f"Input tensor to be quantized should be numpy array. But got {type(x)}"
        )
    if x_dtype not in ("float4_e2m1", "float4_e1m2"):
        raise RuntimeError(
            f"Dtype of input tensor to be quantized is not supported: {x_dtype}"
        )
    if y_dtype_str not in ("float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {y_dtype_str} yet!")

    max_offset = _get_max_offset(x_dtype, y_dtype_str)

    expend_flag = False
    if x.ndim == 2:
        expend_flag = True
        x = numpy.expand_dims(x, axis=0)
        mxscale = numpy.expand_dims(mxscale, axis=0)

    BLOCK_SIZE = 32
    batch, rows, cols = x.shape
    col_blocks = (((cols + BLOCK_SIZE - 1) // BLOCK_SIZE) + 1) // 2 * 2
    row_blocks = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE

    mxscale_exp = _decode_e8m0_to_exp(mxscale)
    mxscale_exp = _unpack_col_blocks(mxscale_exp, col_blocks)

    pad_rows = row_blocks * BLOCK_SIZE - rows
    if pad_rows > 0:
        mxscale_exp_padded = numpy.pad(
            mxscale_exp,
            ((0, 0), (0, pad_rows), (0, 0)),
            mode="constant",
            constant_values=E8M0_EXP_MIN,
        )
    else:
        mxscale_exp_padded = mxscale_exp
    mxscale_exp_blocks = mxscale_exp_padded.reshape(
        batch, row_blocks, BLOCK_SIZE, col_blocks
    )
    pre_broadcast_scale_exp = numpy.max(mxscale_exp_blocks, axis=2) - max_offset

    scale1_exp = numpy.repeat(pre_broadcast_scale_exp, BLOCK_SIZE, axis=1)[:, :rows, :]
    scale2_exp = numpy.repeat(pre_broadcast_scale_exp, BLOCK_SIZE, axis=2)[:, :, :cols]

    mxscale_exp_full = numpy.repeat(mxscale_exp, BLOCK_SIZE, axis=2)[:, :, :cols]
    scale2_exp_full = numpy.repeat(scale2_exp, BLOCK_SIZE, axis=1)[:, :rows, :]
    quant_scale_exp_full = mxscale_exp_full - scale2_exp_full
    quant_scale_exp_full = numpy.where(
        quant_scale_exp_full < -127, -127.0, quant_scale_exp_full
    )

    x_float = x.astype(numpy.float32)
    y_float = x_float * (2.0**quant_scale_exp_full)

    y_dtype_np = _numpy_mx_dtype(y_dtype_str)
    y = y_float.astype(y_dtype_np, copy=False)

    scale1 = _encode_exp_to_e8m0(_pack_scale1(scale1_exp, col_blocks))
    scale2 = _encode_exp_to_e8m0(_pack_scale2(scale2_exp, row_blocks))

    if expend_flag:
        y = numpy.squeeze(y, axis=0)
        scale1 = numpy.squeeze(scale1, axis=0)
        scale2 = numpy.squeeze(scale2, axis=0)
    return y, scale1, scale2


def mx_to_block_mx_quant_golden(x, mxscale, *, dst_type=36, **kwargs):
    """
    Golden function for mx_to_block_mx_quant.
    All the parameters (names and order) follow @mx_to_block_mx_quant_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensors: y, scale1, scale2
    """
    y_dtype_str = DATA_TYPE_INT_TO_STR[dst_type]

    ret = _mx_to_block_mx_quant(
        x,
        mxscale,
        y_dtype_str=y_dtype_str,
    )

    return ret[0], ret[1], ret[2]
