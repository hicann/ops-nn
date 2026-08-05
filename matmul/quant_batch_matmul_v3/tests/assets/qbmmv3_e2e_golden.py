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
"""E2E 层 golden 实现。

委托 ACLNN golden 做核心计算。
对应 torch API: torch_npu.npu_quant_matmul
"""

import os
import sys


sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import qbmmv3_aclnn_golden as _aclnn


# ============================================================================
# helpers
# ============================================================================

# torch ScalarType enum → dtype 字符串
# npu_quant_matmul 的 output_dtype 参数使用 torch ScalarType int（不是 CANN dtype int）
_SCALAR_TYPE_TO_STR = {
    1: "int8",
    3: "int32",
    5: "float16",
    6: "float32",
    15: "bfloat16",
}

# torch_npu dtype enum → dtype 字符串
# npu_quant_matmul 的 x1_dtype/x2_dtype 参数使用 torch_npu dtype int
_TORCH_NPU_DTYPE_TO_STR = {
    290: "hifloat8",
    291: "float8_e5m2",
    292: "float8_e4m3fn",
}


def _output_dtype_to_str(output_dtype):
    """将 output_dtype 参数（int 或 torch.dtype）转换为 dtype 字符串。

    int 值使用 torch ScalarType enum（不是 CANN dtype enum）。
    """
    if output_dtype is None:
        return None
    if isinstance(output_dtype, int):
        return _SCALAR_TYPE_TO_STR.get(output_dtype, str(output_dtype))
    # torch.dtype
    return _util.torch_dtype_to_str(output_dtype)


def _group_sizes_to_int(group_sizes):
    """将 group_sizes [gsM, gsN, gsK] 编码为 int64。

    编码格式: (gsM << 32) | (gsN << 16) | gsK
    对齐 tiling 代码的 group_size 属性编码。
    """
    if group_sizes is None:
        return 0
    if isinstance(group_sizes, (list, tuple)):
        gs_m = int(group_sizes[0]) if len(group_sizes) > 0 else 0
        gs_n = int(group_sizes[1]) if len(group_sizes) > 1 else 0
        gs_k = int(group_sizes[2]) if len(group_sizes) > 2 else 0
        return (gs_m << 32) | (gs_n << 16) | gs_k
    return int(group_sizes)


# ============================================================================
# TestSpec class
# ============================================================================


class TorchNpuQuantMatmulTestSpec:
    """torch_npu.npu_quant_matmul E2E golden。

    参数签名对齐 torch_npu.npu_quant_matmul schema:
      (x1, x2, scale, *, offset, pertoken_scale, bias,
       output_dtype, x1_dtype, x2_dtype, pertoken_scale_dtype,
       scale_dtype, group_sizes, y_scale) -> Tensor

    transpose 从 tensor stride 推断（detect_transpose_from_strides）。
    output_dtype 从参数值推断（CANN int → dtype str）。
    x1_dtype/x2_dtype 等从 tensor dtype 获取，不使用单独参数。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        **kwargs,
    ):
        # 1. 从 tensor stride 推断 transpose
        transpose_x1 = _util.detect_transpose_from_strides(x1)
        transpose_x2 = _util.detect_transpose_from_strides(x2)

        # 2. output_dtype → dtype 字符串
        out_dtype = _output_dtype_to_str(output_dtype)

        # 3. group_sizes → group_size int
        group_size = _group_sizes_to_int(group_sizes)

        # 4. 委托 ACLNN _kernel_qbmm（绕过 golden() 的 _out_dtype fallback）
        # E2E 模式无 out tensor，output dtype 从 output_dtype 参数获取
        x1_np, x2_np, scale_np, offset_np, pertoken_np, bias_np = _aclnn._to_np(
            x1, x2, scale, offset, pertoken_scale, bias
        )

        x2_np = _aclnn._nz_to_nd_if_needed(x2_np, 1, kwargs)
        scale_np = _aclnn._nz_to_nd_if_needed(scale_np, 2, kwargs)

        deq_scale, u64_offset = _aclnn._decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        # 从 x1/x2 shape 推断 output shape（用于 determine_quant_mode 的 M/N）
        if transpose_x1:
            m = x1_np.shape[-1]
        else:
            m = x1_np.shape[-2]
        if transpose_x2:
            n = x2_np.shape[-2]
        else:
            n = x2_np.shape[-1]
        out_shapes = ((m, n),) if x1_np.ndim >= 2 else ()

        return _aclnn._kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            offset_np,
            bias_np,
            pertoken_np,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        *,
        offset=None,
        pertoken_scale=None,
        bias=None,
        output_dtype=None,
        x1_dtype=None,
        x2_dtype=None,
        pertoken_scale_dtype=None,
        scale_dtype=None,
        group_sizes=None,
        y_scale=None,
        **kwargs,
    ):
        """E2E 模式的 customize_inputs（in-place 修改）。

        与 ACLNN customize_inputs 逻辑一致：
        1. E8M0 NaN 清洗
        2. UINT64 scale 合理化重新生成
        """
        _aclnn._customize_inputs_impl(
            x1, x2, scale, offset, bias, pertoken_scale, **kwargs
        )


__spec__ = {
    "torch_npu.npu_quant_matmul": "TorchNpuQuantMatmulTestSpec",
}
