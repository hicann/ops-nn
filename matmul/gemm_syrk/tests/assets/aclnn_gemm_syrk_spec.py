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
GemmSyrk golden spec for TTK aclnn path (ops-test-kit, aclnnGemmSyrk C API direct invoke).

aclnnGemmSyrkGetWorkspaceSize(a, cRef, alphaOptional, betaOptional, transposeX, fillMode, ws, exec):
    C = alpha * (A @ A^T) + beta * C (transposeX=false, a is (m, k)),
    C = alpha * (A^T @ A) + beta * C (transposeX=true, a is the transposed (k, m) storage),
    in-place on cRef (a and cRef share one device buffer).

TTK aclnn calling convention (AclnnParamPlan.build_args):
    - customize_inputs / golden receive parameters POSITIONALLY in C header
      order: (a, cRef, alpha, beta, transposeX, fillMode) — tensors, then scalars,
      then non-tensor params filled from CSV attributes; use *args to capture and
      avoid the context kwargs collision.
    - Tensors arrive as torch.Tensor (fp16/bf16 are torch-native dtypes).
    - customize_inputs modifies tensors in place and returns nothing.
    - golden returns a list of output tensors.

The input c must be symmetric per the syrk contract (the JI region of the
result is written as the transposed mirror of the IJ region), so the input is
symmetrized in-place before the launch. The golden is computed in float32
(matching the kernel's fp32 accumulator) and cast back to the input dtype.
"""

import torch

__spec__ = {"aclnnGemmSyrk": "AclnnGemmSyrkTestSpec"}


def _to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def _matmul_syrk(a_t, transpose_x):
    # transpose_x=false: a is (..., m, k) -> C = a @ a^T
    # transpose_x=true:  a is (..., k, m) -> C = a^T @ a
    if transpose_x:
        return torch.matmul(a_t.transpose(-1, -2), a_t)
    return torch.matmul(a_t, a_t.transpose(-1, -2))


class AclnnGemmSyrkTestSpec:
    @staticmethod
    def customize_inputs(*args, **kwargs):
        # Positional args in C header order: (a, cRef, alpha, beta, transposeX, fillMode).
        c = args[1]
        c_t = _to_torch(c)
        c_sym = ((c_t.float() + c_t.float().transpose(-1, -2)) / 2).to(c_t.dtype)
        if isinstance(c, torch.Tensor):
            c.copy_(c_sym)
        else:
            c[...] = c_sym.numpy()

    @staticmethod
    def golden(*args, **kwargs):
        a, c = _to_torch(args[0]), _to_torch(args[1])
        alpha = (
            float(args[2])
            if len(args) > 2 and args[2] is not None
            else float(kwargs.get("alpha", 1.0))
        )
        beta = (
            float(args[3])
            if len(args) > 3 and args[3] is not None
            else float(kwargs.get("beta", 1.0))
        )
        transpose_x = (
            bool(args[4])
            if len(args) > 4 and args[4] is not None
            else bool(kwargs.get("transposeX", False))
        )
        a_t = a.float()
        c_t = c.float()
        out = alpha * _matmul_syrk(a_t, transpose_x) + beta * c_t
        return [out.to(a.dtype)]
