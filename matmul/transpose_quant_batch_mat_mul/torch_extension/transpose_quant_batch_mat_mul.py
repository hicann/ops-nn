# -*- coding: utf-8 -*-
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

# Supported torch-level input dtypes: MXFP8 (float8_e4m3fn), FP8 (float8_e4m3fn with
# float scale), and HIFP8 (via acl dtype override). MXFP4 uses acl dtype override.
_FP8_E4M3FN_DTYPE = getattr(torch, "float8_e4m3fn", None)
_FP8_E5M2_DTYPE = getattr(torch, "float8_e5m2", None)
_FP8_TORCH_DTYPES = tuple(
    d for d in (_FP8_E4M3FN_DTYPE, _FP8_E5M2_DTYPE) if isinstance(d, torch.dtype)
)

# ACL dtype codes for dtype overrides (DType enum in aclnn_common.h = ACL code + g_toAclOffset).
# Use torch_npu int attrs when available, fall back to the codes from acl_base_rt.h.
_ACL_OFFSET = 256
_ACL_CODE_FLOAT4_E2M1 = 40
_ACL_CODE_FLOAT8_E5M2 = 35
_ACL_CODE_FLOAT8_E4M3FN = 36
_ACL_CODE_HIFLOAT8 = 34


def _acl_override_code(attr_name, acl_code):
    value = getattr(torch_npu, attr_name, None)
    if not isinstance(value, int):
        return _ACL_OFFSET + acl_code
    return value if value >= _ACL_OFFSET else value + _ACL_OFFSET


_FP4_E2M1_ACL = _acl_override_code("float4_e2m1fn_x2", _ACL_CODE_FLOAT4_E2M1)
_FP8_E5M2_ACL = _acl_override_code("float8_e5m2", _ACL_CODE_FLOAT8_E5M2)
_FP8_E4M3FN_ACL = _acl_override_code("float8_e4m3fn", _ACL_CODE_FLOAT8_E4M3FN)
_HIFLOAT8_ACL = _acl_override_code("hifloat8", _ACL_CODE_HIFLOAT8)

# All supported acl dtype codes (int values for x1_dtype/x2_dtype overrides)
_SUPPORTED_ACL_DTYPES = tuple(
    d for d in (_FP4_E2M1_ACL, _FP8_E5M2_ACL, _FP8_E4M3FN_ACL, _HIFLOAT8_ACL)
)


def _normalize_acl_dtype(dtype_code):
    """Accept both the DType enum code (ACL + 256) and the raw ACL code."""
    if isinstance(dtype_code, int) and dtype_code < _ACL_OFFSET:
        return dtype_code + _ACL_OFFSET
    return dtype_code


# dtype attr (int) -> torch output dtype, matching op_api aclnn output rules.
_DTYPE_TO_TORCH = {1: torch.float16, 27: torch.bfloat16, 34: torch.uint8}


def _check_supported_input(x, name, acl_dtype=None):
    """Check that x1/x2 is a supported input dtype (MX fp4/fp8, FP8, or HIFP8)."""
    if x.dtype in _FP8_TORCH_DTYPES:
        return
    if (
        acl_dtype is not None
        and _normalize_acl_dtype(acl_dtype) in _SUPPORTED_ACL_DTYPES
    ):
        return
    raise NotImplementedError(
        "%s torch interface supports fp8 (float8_e4m3fn/float8_e5m2), hifp8, and MX fp4/fp8 input, "
        "got dtype %s (acl dtype %s)" % (name, x.dtype, acl_dtype)
    )


def _output_dtype(dtype: int) -> torch.dtype:
    out_dtype = _DTYPE_TO_TORCH.get(dtype)
    if out_dtype is None:
        raise NotImplementedError(
            "unsupported output dtype %s; only float16 (1), bfloat16 (27) and hifloat8 (34) are supported"
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
            # Reject unsupported inputs at meta time as well.
            _check_supported_input(x1, "x1", x1_dtype)
            _check_supported_input(x2, "x2", x2_dtype)
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
    # Torch entry accepts fp8, hifp8, and MX fp4/fp8 inputs. The csrc validates
    # x1/x2 against the resolved acl dtypes (including x1_dtype/x2_dtype overrides).
    _check_supported_input(x1, "x1", x1_dtype)
    _check_supported_input(x2, "x2", x2_dtype)

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
