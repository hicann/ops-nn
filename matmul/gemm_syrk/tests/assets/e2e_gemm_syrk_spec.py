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
GemmSyrk golden spec for TTK E2E path (ops-test-kit, torch direct invoke).

api_name: torch.ops.cann_ops_nn.gemm_syrk (cann_ops_nn torch extension, in-place on c).

    C = alpha * (A @ A^T) + beta * C (transpose_x=False, a is (m, k)),
    C = alpha * (A^T @ A) + beta * C (transpose_x=True, a is the transposed (k, m) storage).

The extension is registered by the cann_ops_nn package (torch_extension/); TTK's
TorchOpsPackageLoader imports cann_ops_nn on demand when resolving
"torch.ops.cann_ops_nn.gemm_syrk".

TTK e2e calling convention (ParamPlan.build_args):
    - customize_inputs / golden receive (a, c) positionally, alpha/beta/transpose_x/
      fill_mode as kwargs (keyword-only schema params fed from CSV attributes).
    - Tensors arrive as CPU torch.Tensor.
    - customize_inputs modifies tensors in place and returns nothing.
    - golden returns a list of output tensors.

The input c must be symmetric per the syrk contract (the JI region of the
result is written as the transposed mirror of the IJ region), so the input is
symmetrized in-place before the launch. The golden is computed in float32
(matching the kernel's fp32 accumulator) and cast back to the input dtype.
"""

import torch

__spec__ = {"torch.ops.cann_ops_nn.gemm_syrk": "E2eGemmSyrkTestSpec"}


def _matmul_syrk(a_t, transpose_x):
    if transpose_x:
        return torch.matmul(a_t.transpose(-1, -2), a_t)
    return torch.matmul(a_t, a_t.transpose(-1, -2))


class E2eGemmSyrkTestSpec:
    @staticmethod
    def customize_inputs(a, c, **kwargs):
        # Positional args: (a, c); alpha/beta/transpose_x/fill_mode arrive as kwargs.
        c_sym = ((c.float() + c.float().transpose(-1, -2)) / 2).to(c.dtype)
        c.copy_(c_sym)

    @staticmethod
    def golden(
        a, c, alpha=1.0, beta=1.0, transpose_x=False, fill_mode="full", **kwargs
    ):
        a_t = a.float()
        c_t = c.float()
        out = alpha * _matmul_syrk(a_t, transpose_x) + beta * c_t
        return [out.to(a.dtype)]
