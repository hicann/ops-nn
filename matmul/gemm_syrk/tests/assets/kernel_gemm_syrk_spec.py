# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
GemmSyrk golden spec for TTK (ops-test-kit, kernel pathway).

GemmSyrk: C = alpha * (A @ A^T) + beta * C   (transpose_x = False, a is (m, k))
          C = alpha * (A^T @ A) + beta * C   (transpose_x = True, a is the transposed (k, m) storage)
  - a: (m, k) or (batch, m, k) / (k, m) or (batch, k, m) when transposed, fp16/bf16, ND
  - c: (m, m) or (batch, m, m), symmetric input, in-place input/output

The kernel pathway feeds the golden numpy arrays (bfloat16 arrives as
ml_dtypes.bfloat16, which torch.from_numpy cannot ingest — bridge through the
bit-identical uint16 view). The golden computes in float32 (matching the fp32
accumulator of the kernel) and casts back to the input dtype. Attributes
(alpha, beta, transpose_x, fill_mode) arrive as keyword arguments.
"""

import numpy
import torch
from ml_dtypes import bfloat16 as ml_bfloat16

__spec__ = {"gemm_syrk": "GemmSyrkTestSpec"}


def _np_to_torch(x):
    x = numpy.ascontiguousarray(x)
    if x.dtype == ml_bfloat16:
        # ml_dtypes.bfloat16 is bit-identical to torch.bfloat16 (2 bytes).
        return torch.from_numpy(x.view(numpy.uint16)).view(torch.bfloat16)
    return torch.from_numpy(x)


def _torch_to_np_like(t, ref):
    ref_np = numpy.asarray(ref)
    if ref_np.dtype == ml_bfloat16:
        return t.contiguous().view(torch.uint16).numpy().view(ml_bfloat16)
    return t.numpy()


def _matmul_syrk(a_t, transpose_x):
    if transpose_x:
        return torch.matmul(a_t.transpose(-1, -2), a_t)
    return torch.matmul(a_t, a_t.transpose(-1, -2))


def gemm_syrk_golden(
    a, c, alpha=1.0, beta=1.0, transpose_x=False, fill_mode="full", **kwargs
):
    a_t = _np_to_torch(a).float()
    c_t = _np_to_torch(c).float()
    out = alpha * _matmul_syrk(a_t, transpose_x) + beta * c_t
    out = out.to(_np_to_torch(a).dtype)
    return [_torch_to_np_like(out, a)]


def gemm_syrk_customize_inputs(a, c, **kwargs):
    # The GemmSyrk contract requires c to be a symmetric matrix (the JI region
    # of the result is written as the transposed mirror of the IJ region).
    # Symmetrize the randomly generated c to satisfy it: c = (c + c^T) / 2.
    c_t = numpy.asarray(c)
    c_sym = (c_t + numpy.swapaxes(c_t, -1, -2)) / 2
    return (a, numpy.ascontiguousarray(c_sym))


class GemmSyrkTestSpec:
    golden = gemm_syrk_golden
    customize_inputs = gemm_syrk_customize_inputs
