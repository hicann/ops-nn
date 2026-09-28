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
"""E2E-level golden for torch_npu.npu_transpose_quant_batchmatmul.

Delegates core computation to the ACLNN-level golden; this layer only performs
torch API parameter mapping:

- dtype   : torch enum (5 = float16, 15 = bfloat16) -> output dtype string
- group_sizes : [gm, gn, gk] packed into the group_size attr
- x1_dtype / x2_dtype : optional "carrier dtype" declarations - when a tensor
  is passed as a uint8 carrier, reinterpret its bit pattern as the declared
  NPU dtype (ge.Bitcast semantics, e.g. enum 290 = hifloat8).

All quantization semantics live in the kernel-level golden.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import tqbmm_aclnn_golden as _aclnn

# torch dtype enum -> golden output dtype string (torch_npu enum table,
# _meta_registrations.py TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP).
_TORCH_DTYPE_TO_STR = {
    5: "float16",
    15: "bfloat16",
    290: "hifloat8",
}

# torch_npu private dtype enum -> numpy view target for uint8 carriers.
# Sources: torch_npu _meta_registrations.py enum table (24/290); 23/24 are
# defensive entries - they have native torch scalar types and normally never
# travel through the carrier path.
_NPU_DTYPE_TO_NP_VIEW = {
    23: _util.np_fp8_e5m2,
    24: _util.np_fp8_e4m3,
    290: _util.np_hif8,
}


def _group_sizes_pack(group_sizes):
    """[gm, gn, gk] -> (gm << 32) | (gn << 16) | gk (tiling group_size packing)."""
    gm, gn, gk = (list(group_sizes) + [0, 0, 0])[:3] if group_sizes else (0, 0, 0)
    return (gm << 32) | (gn << 16) | gk


def _apply_npu_dtype_view(tensor, npu_dtype, arg_name):
    """Reinterpret a uint8 carrier tensor as its declared NPU dtype.

    Mirrors the torch_npu ge converter: when x1_dtype/x2_dtype are not None,
    the tensor is bit-cast to the declared dtype (ge.Bitcast semantics).
    """
    if npu_dtype is None or tensor is None:
        return tensor
    np_dtype = _NPU_DTYPE_TO_NP_VIEW.get(int(npu_dtype))
    if np_dtype is None:
        raise NotImplementedError(
            f"Unsupported {arg_name} npu dtype enum: {npu_dtype} "
            f"(supported: {sorted(_NPU_DTYPE_TO_NP_VIEW)})"
        )
    if isinstance(tensor, np.ndarray):
        carrier = tensor
    else:
        carrier = _util.torch_to_numpy(tensor)
    if carrier.dtype != np.uint8:
        raise ValueError(
            f"{arg_name} is declared as npu dtype {npu_dtype}, the tensor must be "
            f"a uint8 carrier, got {carrier.dtype}"
        )
    return np.ascontiguousarray(carrier).view(np_dtype)


class TorchNpuNpuTransposeQuantBatchmatmulTestSpec:
    compare = _util.isclose_compare

    @staticmethod
    def golden(
        x1,
        x2,
        dtype,
        *,
        bias=None,
        x1_scale=None,
        x2_scale=None,
        group_sizes=None,
        perm_x1=None,
        perm_x2=None,
        perm_y=None,
        batch_split_factor=1,
        x1_dtype=None,
        x2_dtype=None,
        **kwargs,
    ):
        out_dtype = _TORCH_DTYPE_TO_STR.get(int(dtype) if dtype is not None else None)
        if out_dtype is None:
            raise NotImplementedError(
                f"Unsupported output dtype enum: {dtype} "
                f"(supported: {sorted(_TORCH_DTYPE_TO_STR)})"
            )

        x1 = _apply_npu_dtype_view(x1, x1_dtype, "x1_dtype")
        x2 = _apply_npu_dtype_view(x2, x2_dtype, "x2_dtype")

        temp_kwargs = dict(kwargs)
        temp_kwargs["output_dtypes"] = [out_dtype]

        return _aclnn.AclnnTransposeQuantBatchMatMulTestSpec.golden(
            x1,
            x2,
            bias=bias,
            x1Scale=x1_scale,
            x2Scale=x2_scale,
            dtype=0,
            groupSize=_group_sizes_pack(group_sizes),
            permX1=perm_x1,
            permX2=perm_x2,
            permY=perm_y,
            batchSplitFactor=batch_split_factor,
            out=None,
            **temp_kwargs,
        )


__spec__ = {
    "torch_npu.npu_transpose_quant_batchmatmul": "TorchNpuNpuTransposeQuantBatchmatmulTestSpec",
}
