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

"""Dequantize golden（kernel / geir 通路）。

数值逐点对齐 op_host/arch35/dequantize_tiling_arch35.cpp 的常量与
op_kernel/arch35 三个 mode kernel 的 VF 链（全程 float32）：

- MIN_COMBINED（README 公式）:
      y = (x + bias) * (max_range - min_range) / denom + min_range
  常量与 tiling 一致：int8  denom=255、bias=128；uint8 denom=255、bias=0；
  int32 denom=2**32、bias=2**31（tiling kDenomInt32 为 2^32，README 的
  2^n - 1 记法对 int32 与 kernel 有 1 的出入，以 kernel/tiling 为准）。
- MIN_FIRST: 先算 MIN_COMBINED，再减去偏置
      q = round_half_away(min_range * (-denom) / max(max_range - min_range, 1e-30))
      y = y - (min_range + q * scale)
  仅 int8/uint8 生效；int32 与 MIN_COMBINED 完全一致（kernel 对 int32 跳过
  offset 阶段）。max_range == min_range 时 kernel Select 输出 min_range。
  round 采用 AscendC RoundMode::CAST_ROUND（round half away from zero，
  经 int32 中转），非 numpy 的 banker 舍入。
- SCALED: y = x * (max_range / denom)，不使用 min_range。
  denom：int8=127（2^7-1）、uint8=255（2^8-1）、int32=2**31
  （tiling kDenomScaledInt32，README 的 2^(n-1)-1 记法同样以 tiling 为准）。

min_range / max_range 按 numpy 广播规则对齐 x（README：支持广播）。
输出恒为 float32（dequantize_def.cpp 仅声明 DT_FLOAT 输出）。
"""

import numpy as np

__golden__ = {"kernel": {"dequantize": "dequantize_golden"}}

# 与 dequantize_tiling_arch35.cpp 的 kDenom*/kBias* 常量一一对应
_MIN_COMBINED_DENOM = {"int8": 255.0, "uint8": 255.0, "int32": 4294967296.0}
_MIN_COMBINED_BIAS = {"int8": 128.0, "uint8": 0.0, "int32": 2147483648.0}
_SCALED_DENOM = {"int8": 127.0, "uint8": 255.0, "int32": 2147483648.0}
_SUPPORTED_MODES = ("MIN_COMBINED", "MIN_FIRST", "SCALED")


def _round_half_away(values):
    """AscendC RoundMode::CAST_ROUND：round half away from zero，经 int32 中转。

    超出 int32 的值按饱和处理（与硬件整型转换一致）；±inf/NaN 仅出现在
    zero-range 等最终被 Select 丢弃的路径上，这里不产生告警。
    """
    floats = np.asarray(values, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.where(
            floats >= np.float32(0.0),
            np.floor(floats + np.float32(0.5)),
            np.ceil(floats - np.float32(0.5)),
        )
        rounded = np.clip(rounded, np.float32(-2147483648.0), np.float32(2147483647.0))
        return rounded.astype(np.int32).astype(np.float32)


def dequantize_golden(x, min_range, max_range, *, mode="MIN_COMBINED", **kwargs):
    """
    Golden function for dequantize.
    All the parameters (names and order) follow @dequantize_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        x: quantized int8/uint8/int32 tensor
        min_range: float32 tensor broadcast-compatible with x
        max_range: float32 tensor broadcast-compatible with x (must be > min_range)
        mode: MIN_COMBINED / MIN_FIRST / SCALED (default MIN_COMBINED per def)
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (float32 numpy array, same shape as x)
    """
    x = np.asarray(x)
    min_range = np.asarray(min_range, dtype=np.float32)
    max_range = np.asarray(max_range, dtype=np.float32)

    mode_name = "MIN_COMBINED" if mode is None else str(mode).upper()
    if mode_name not in _SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported dequantize mode: {mode_name!r}, expected one of {_SUPPORTED_MODES}"
        )

    dtype_name = x.dtype.name
    if dtype_name not in _MIN_COMBINED_DENOM:
        raise ValueError(
            f"Unsupported x dtype for dequantize: {dtype_name!r}, expected int8/uint8/int32"
        )

    # 与 kernel 相同的整型->float32 转换（int8/uint8 两步转换无损，int32 就近舍入）
    xf = x.astype(np.float32)

    if mode_name == "SCALED":
        # VF 链: Div(max_range, denom)->scale -> Mul(x, scale)；min_range 不参与
        scale = max_range / np.float32(_SCALED_DENOM[dtype_name])
        y = xf * scale
    else:
        denom = np.float32(_MIN_COMBINED_DENOM[dtype_name])
        bias = np.float32(_MIN_COMBINED_BIAS[dtype_name])
        # VF 链: Sub(max,min)->range -> Div(range,denom)->scale
        #        -> Adds(bias) -> Mul(scale) -> Add(min)
        value_range = max_range - min_range
        scale = value_range / denom
        y = (xf + bias) * scale + min_range
        if mode_name == "MIN_FIRST" and dtype_name in ("int8", "uint8"):
            # int32 的 MIN_FIRST 与 MIN_COMBINED 完全一致，kernel 跳过 offset 阶段
            # pre: range 先夹到 1e-30 再算 q = round(min * (-denom) / range)
            range_safe = np.maximum(value_range, np.float32(1e-30))
            q = _round_half_away(min_range * (-denom) / range_safe)
            # post: offset = min + q * scale; y = y - offset
            y = y - (min_range + q * scale)
            # zero-range (max == min) 时 kernel Select 直接输出 min_range
            y = np.where(value_range == np.float32(0.0), min_range, y)

    output_dtypes = kwargs.get("output_dtypes") or ["float32"]
    return y.astype(output_dtypes[0])
