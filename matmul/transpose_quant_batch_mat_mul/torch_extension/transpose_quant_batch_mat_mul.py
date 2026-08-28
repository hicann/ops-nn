# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional, List
import torch
import torch_npu
from torch.library import impl
from cann_ops_nn.op_builder import OpBuilder, get_as_library

# The torch-level interface only serves the MX path: MXFP8 (float8_e4m3fn with
# e8m0 scale) and MXFP4 (float4_e2m1 with e8m0 scale).  FP8-INT8 / Hifp8 inputs
# must be rejected here so the kernel (or the graph-mode infer dtype below) does
# not fail with a confusing downstream error.
_FP8_E4M3FN_DTYPE = getattr(torch, "float8_e4m3fn", None)
_MX_TORCH_DTYPES = tuple(d for d in (_FP8_E4M3FN_DTYPE,) if isinstance(d, torch.dtype))
_FP4_E2M1_ACL = getattr(torch_npu, "float4_e2m1fn_x2", None)
_FP8_E4M3FN_ACL = getattr(torch_npu, "float8_e4m3fn", None)
_MX_ACL_DTYPES = tuple(
    d for d in (_FP4_E2M1_ACL, _FP8_E4M3FN_ACL) if isinstance(d, int)
)

# dtype attr (int) -> torch output dtype, matching op_api aclnn output rules.
_DTYPE_TO_TORCH = {1: torch.float16, 27: torch.bfloat16}


def _check_mx_input(x, name, acl_dtype=None):
    if x.dtype in _MX_TORCH_DTYPES:
        return
    if acl_dtype is not None and acl_dtype in _MX_ACL_DTYPES:
        return
    raise NotImplementedError(
        "%s torch interface only supports MX fp4/fp8 (float4_e2m1 or "
        "float8_e4m3fn) input, got dtype %s (acl dtype %s)" % (name, x.dtype, acl_dtype)
    )


def _output_dtype(dtype: int) -> torch.dtype:
    out_dtype = _DTYPE_TO_TORCH.get(dtype)
    if out_dtype is None:
        raise NotImplementedError(
            "unsupported output dtype %s; only float16 (1) and bfloat16 (27) are supported"
            % dtype
        )
    return out_dtype


class TransposeQuantBatchMatMulOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("transpose_quant_batch_mat_mul")

    def sources(self) -> list:
        return [self.resolve_source("transpose_quant_batch_mat_mul.cpp")]

    def schema(self) -> str:
        return (
            "transpose_quant_batch_mat_mul("
            "Tensor x1, Tensor x2, *, int dtype, Tensor? bias=None, Tensor? x1_scale=None, "
            "Tensor? x2_scale=None, int[]? group_sizes=None, int[]? perm_x1=None, "
            "int[]? perm_x2=None, int[]? perm_y=None, int? batch_split_factor=None, "
            "int? x1_dtype=None, int? x2_dtype=None, "
            "int? x1_scale_dtype=None, int? x2_scale_dtype=None"
            ") -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def transpose_quant_batch_mat_mul_meta(
            x1: torch.Tensor,
            x2: torch.Tensor,
            *,
            dtype: int,
            bias: Optional[torch.Tensor] = None,
            x1_scale: Optional[torch.Tensor] = None,
            x2_scale: Optional[torch.Tensor] = None,
            group_sizes: Optional[List[int]] = None,
            perm_x1: Optional[List[int]] = None,
            perm_x2: Optional[List[int]] = None,
            perm_y: Optional[List[int]] = None,
            batch_split_factor: Optional[int] = None,
            x1_dtype: Optional[int] = None,
            x2_dtype: Optional[int] = None,
            x1_scale_dtype: Optional[int] = None,
            x2_scale_dtype: Optional[int] = None,
        ) -> torch.Tensor:
            # Reject FP8-INT8 / Hifp8 inputs at meta time as well.
            _check_mx_input(x1, "x1", x1_dtype)
            _check_mx_input(x2, "x2", x2_dtype)
            out_dtype = _output_dtype(dtype)

            default_perm_x1 = [1, 0, 2]
            default_perm_x2 = [0, 1, 2]

            perm_x1_real = perm_x1 if perm_x1 is not None else default_perm_x1
            perm_x2_real = perm_x2 if perm_x2 is not None else default_perm_x2
            batch_split_factor_value = (
                batch_split_factor if batch_split_factor is not None else 1
            )

            x1_is_fp4 = x1_dtype == _FP4_E2M1_ACL
            x2_is_fp4 = x2_dtype == _FP4_E2M1_ACL
            x1_last_dim = x1.dim() - 1
            x2_last_dim = x2.dim() - 1

            m_dim = x1.size(perm_x1_real[1])
            if x1_is_fp4 and perm_x1_real[1] == x1_last_dim:
                m_dim *= 2
            batch_dim = x1.size(perm_x1_real[0])
            n_dim = x2.size(perm_x2_real[2])
            if x2_is_fp4 and perm_x2_real[2] == x2_last_dim:
                n_dim *= 2

            output_size = [m_dim, batch_dim, n_dim]

            if batch_split_factor_value > 1:
                output_size = [
                    batch_split_factor_value,
                    m_dim,
                    batch_dim * n_dim // batch_split_factor_value,
                ]

            return torch.empty(output_size, dtype=out_dtype, device="meta")


transpose_quant_batch_mat_mul_builder = TransposeQuantBatchMatMulOpBuilder()
transpose_quant_batch_mat_mul_builder._ensure_initialized()


@impl(get_as_library(), transpose_quant_batch_mat_mul_builder.name, "PrivateUse1")
def transpose_quant_batch_mat_mul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    *,
    dtype: int,
    bias: Optional[torch.Tensor] = None,
    x1_scale: Optional[torch.Tensor] = None,
    x2_scale: Optional[torch.Tensor] = None,
    group_sizes: Optional[List[int]] = None,
    perm_x1: Optional[List[int]] = None,
    perm_x2: Optional[List[int]] = None,
    perm_y: Optional[List[int]] = None,
    batch_split_factor: Optional[int] = None,
    x1_dtype: Optional[int] = None,
    x2_dtype: Optional[int] = None,
    x1_scale_dtype: Optional[int] = None,
    x2_scale_dtype: Optional[int] = None,
) -> torch.Tensor:
    # Torch entry rejects FP8-INT8 / Hifp8 input up-front; only MX fp4/fp8
    # (with e8m0 scale) is exposed.  The csrc validates x1/x2 against the
    # resolved acl dtypes (including the x1_dtype/x2_dtype overrides).
    _check_mx_input(x1, "x1", x1_dtype)
    _check_mx_input(x2, "x2", x2_dtype)

    op_module_matmul = transpose_quant_batch_mat_mul_builder.load()
    return op_module_matmul.transpose_quant_batch_mat_mul(
        x1,
        x2,
        dtype,
        bias,
        x1_scale,
        x2_scale,
        group_sizes,
        perm_x1,
        perm_x2,
        perm_y,
        batch_split_factor,
        x1_dtype,
        x2_dtype,
        x1_scale_dtype,
        x2_scale_dtype,
    )
