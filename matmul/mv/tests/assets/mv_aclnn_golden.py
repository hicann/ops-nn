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
"""ACLNN 层 golden 实现 for aclnnMv。

aclnnMv 语义：self (n x m) @ vec (m,) → out (n,)

计算流程：
  1. torch → numpy
  2. cubeMathType 精度模拟
  3. vec unsqueeze: (m,) → (m, 1)
  4. matmul: (n, m) @ (m, 1) → (n, 1)
  5. squeeze: (n, 1) → (n,)
  6. cast to output dtype
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util


def _to_np(*tensors):
    """torch tensor → numpy。None 保持 None，已是 numpy 则跳过。"""
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        else:
            result.append(_util.torch_to_numpy(t))
    return tuple(result)


def _out_dtype(out, fallback):
    """返回 out tensor 的 dtype 名，若 out 为 None 则用 fallback 的 dtype。"""
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _apply_cube_type(x1, x2, cubeMathType):
    """cubeMathType 精度模拟。

    - mode 2 (USE_FP16): fp32→fp16 截断，模拟 NPU fp16 计算
    - modes 1/3 (HF32): fp32→hf32 截断（19-bit mantissa）
    - mode 0 (KEEP_DTYPE): 无操作
    """
    if cubeMathType is None:
        return x1, x2
    cmt = int(cubeMathType)
    if cmt == 2:
        if x1 is not None and x1.dtype == np.float32:
            x1 = x1.astype(np.float16)
        if x2 is not None and x2.dtype == np.float32:
            x2 = x2.astype(np.float16)
    elif cmt in (1, 3):
        if x1 is not None and x1.dtype == np.float32:
            x1 = _util.hf32_truncate_np(x1)
        if x2 is not None and x2.dtype == np.float32:
            x2 = _util.hf32_truncate_np(x2)
    return x1, x2


class AclnnMvTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(self, vec, out=None, cubeMathType=0, **kwargs):
        """aclnnMv: self (n x m) @ vec (m,) → out (n,)"""
        # 1) torch → numpy
        self_np, vec_np = _to_np(self, vec)

        # 2) 空 tensor 处理：self 为空时返回全 0
        if self_np is None or self_np.size == 0:
            out_dtype = _out_dtype(out, self)
            if out is not None:
                (out_np,) = _to_np(out)
                return np.zeros_like(out_np)
            return np.zeros(self_np.shape[0] if self_np is not None else 0)

        # 3) cubeMathType 精度模拟
        self_np, vec_np = _apply_cube_type(self_np, vec_np, cubeMathType)

        # 4) 核心计算：vec unsqueeze → matmul → squeeze
        #    (n, m) @ (m, 1) → (n, 1) → (n,)
        vec_2d = vec_np[..., np.newaxis]  # (m,) → (m, 1)

        # 5) 精度提升计算 (fp16/bf16 → fp32 累加，匹配 NPU 行为)
        if self_np.dtype in (np.float16, _util.np_bfloat16):
            compute_dtype = np.float32
        else:
            compute_dtype = np.float64

        result = np.matmul(self_np.astype(compute_dtype), vec_2d.astype(compute_dtype))
        result = result.squeeze(-1)  # (n, 1) → (n,)

        # 6) Cast to output dtype
        out_dtype = _out_dtype(out, self)
        return _util.cast_output_dtype(result, out_dtype) if out_dtype else result


__spec__ = {
    "aclnnMv": "AclnnMvTestSpec",
}
