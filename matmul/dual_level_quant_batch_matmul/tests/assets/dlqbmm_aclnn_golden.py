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
"""ACLNN-level golden for aclnnDualLevelQuantMatmulWeightNz."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
from dlqbmm_kernel_golden import DualLevelQuantBatchMatmulTestSpec


def _find_storage(tensor):
    cur = tensor
    best = None
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if (
            isinstance(cur, np.ndarray)
            and cur.flags["C_CONTIGUOUS"]
            and (best is None or cur.size >= best.size)
        ):
            best = cur
        cur = getattr(cur, "base", None)
    return best


def _unpack_int8_to_fp4(storage, storage_shape):
    u8 = storage.view(np.uint8)
    lo = u8 & 0x0F
    hi = (u8 >> 4) & 0x0F
    fp4 = np.stack((lo, hi), axis=-1).view(_util.np_fp4_e2m1)
    return fp4.reshape(storage_shape[:-1] + (64,))


def _decode_nz_x2(x2, ori_shape):
    if not isinstance(x2, np.ndarray):
        raise TypeError(f"x2 must be numpy for the FRACTAL_NZ path, got {type(x2)}")

    storage = _find_storage(x2)
    if storage is None:
        raise RuntimeError("cannot find underlying numpy storage for x2")

    n, k = ori_shape
    if x2.dtype in (np.int8, np.uint8):
        storage_shape = (math.ceil(k / 64), math.ceil(n / 16), 16, 32)
        need = math.prod(storage_shape)
        if storage.size != need:
            raise RuntimeError(
                f"x2 int8 storage size mismatch: got {storage.size}, expected {need}, "
                f"ori_shape={ori_shape}, storage_shape={storage_shape}"
            )
        nz = _unpack_int8_to_fp4(storage, storage_shape)
    else:
        storage_shape = (math.ceil(k / 64), math.ceil(n / 16), 16, 64)
        need = math.prod(storage_shape)
        if storage.size != need:
            raise RuntimeError(
                f"x2 storage size mismatch: got {storage.size}, expected {need}, "
                f"ori_shape={ori_shape}, storage_shape={storage_shape}"
            )
        nz = storage.reshape(storage_shape)
    return _util.nz_to_nd(nz, ori_shape=ori_shape)


class AclnnDualLevelQuantMatmulWeightNzTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        x1Level0Scale,
        x2Level0Scale,
        x1Level1Scale,
        x2Level1Scale,
        optionalBias=None,
        transposeX1=False,
        transposeX2=True,
        level0GroupSize=512,
        level1GroupSize=32,
        out=None,
        **kwargs,
    ):
        formats = kwargs.get("tensor_formats", ())
        if len(formats) > 1 and formats[1] == "FRACTAL_NZ":
            x2_shape = tuple(int(v) for v in x2.shape)
            if x2.dtype in (np.int8, np.uint8):
                ori_shape = (x2_shape[0], x2_shape[1] * 2)
            else:
                ori_shape = x2_shape
            x2 = _decode_nz_x2(x2, ori_shape)

        dtype = int(kwargs.pop("dtype", 1))
        tensor_dtypes = kwargs.get("tensor_dtypes")
        if "output_dtypes" not in kwargs and tensor_dtypes and len(tensor_dtypes) > 7:
            out_dtype = tensor_dtypes[7]
            if out_dtype is not None:
                kwargs["output_dtypes"] = [out_dtype]

        return DualLevelQuantBatchMatmulTestSpec.golden(
            x1,
            x2,
            x1Level0Scale,
            x1Level1Scale,
            x2Level0Scale,
            x2Level1Scale,
            optionalBias,
            dtype=dtype,
            transpose_x1=bool(transposeX1),
            transpose_x2=bool(transposeX2),
            level0_group_size=int(level0GroupSize),
            level1_group_size=int(level1GroupSize),
            **kwargs,
        )


__spec__ = {
    "aclnnDualLevelQuantMatmulWeightNz": "AclnnDualLevelQuantMatmulWeightNzTestSpec"
}
