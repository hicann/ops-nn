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
"""ACLNN 层 golden 实现。

各 API 的 golden 按统一流程组织：
  1. torch → numpy
  2. NZ→ND 格式转换（WeightNz 的 x2 和 scale）
  3. UINT64 scale 解码
  4. dtype 从 out tensor 推断（对应 C++ GetDtypeAndTranspose: dtype = out->GetDataType()）
  5. 调 kernel golden 做核心计算
  6. 返回 numpy 结果

参数名严格对齐 C API GetWorkspaceSize 声明。
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import matmul_golden_util as _util
import qbmmv3_kernel_golden as _kernel
import qbmmv3_scalenz_util as _scalenz


# ============================================================================
# shared helpers
# ============================================================================


def _out_shape(out):
    """从 out tensor 获取 output shape（用于 determine_quant_mode 推断 M/N）。"""
    if out is None:
        return ()
    (out_np,) = _util.to_numpy(out)
    if out_np is not None:
        return (out_np.shape,)
    return ()


def _kernel_qbmm(
    x1_np,
    x2_np,
    scale_np,
    offset_np,
    bias_np,
    pertoken_np,
    *,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype,
    out_shapes,
    **kwargs,
):
    """调 kernel golden 做核心计算，返回 numpy 结果。"""
    temp_kwargs = dict(kwargs)
    if out_dtype:
        temp_kwargs["output_dtypes"] = [out_dtype]
    # uint8 e8m0 载体：e2e 侧已解码数值并给出 scale_dtype 提示，
    # 此处优先使用提示，避免按 uint8 载体 dtype 误判量化模式。
    hint = temp_kwargs.pop("scale_dtype_hint", None)
    if hint:
        temp_kwargs["scale_dtype"] = hint
    elif scale_np is not None:
        temp_kwargs["scale_dtype"] = _util.dtype_to_str(scale_np.dtype)
    if out_shapes:
        temp_kwargs["output_shapes"] = out_shapes
        temp_kwargs["output_ori_shapes"] = out_shapes
    return _kernel.qbmmv3_kernel_compute(
        x1_np,
        x2_np,
        scale_np,
        offset_np,
        bias_np,
        pertoken_np,
        dtype=0,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        group_size=group_size,
        **temp_kwargs,
    )[0]


# ============================================================================
# customize_inputs helpers
# ============================================================================


def _customize_inputs_impl(x1, x2, scale, offset, bias, pertoken_scale, **kwargs):
    """ACLNN 模式的 customize_inputs 核心逻辑（in-place 修改）。

    与 kernel 模式的 customize_inputs 逻辑一致：
    1. E8M0 NaN 清洗
    2. UINT64 scale 合理化重新生成

    ACLNN 模式不捕获返回值，需要用 write_back in-place 修改。
    """
    testcase_name = kwargs.get("testcase_name", "unknown")

    # 1. E8M0 NaN 清洗（in-place write_back）
    for tensor in [x1, x2, scale, pertoken_scale]:
        if tensor is None:
            continue
        cleaned = _util.sanitize_e8m0_scale(tensor, testcase_name)
        if cleaned is not tensor:
            _util.write_back(tensor, cleaned)

    # 1b. uint8/int8 e8m0 carrier：raw 255 是 NaN，按 scale dtype 提示清洗。
    # 只处理 scale/pertoken 槽，避免误清 x1/x2 的 fp4 uint8 数据。
    for tensor, dtype_hint in (
        (scale, kwargs.get("scale_dtype")),
        (pertoken_scale, kwargs.get("pertoken_scale_dtype")),
    ):
        if tensor is None or dtype_hint is None:
            continue
        try:
            is_e8m0 = int(dtype_hint) == 293
        except (TypeError, ValueError):
            is_e8m0 = str(dtype_hint) == "float8_e8m0"
        if not is_e8m0:
            continue
        cleaned = _util.sanitize_e8m0_carrier(tensor, testcase_name)
        if cleaned is not tensor:
            _util.write_back(tensor, cleaned)

    # 2. UINT64 scale 合理化重新生成（in-place write_back）
    if scale is not None:
        if isinstance(scale, np.ndarray):
            scale_dt = _util.dtype_to_str(scale.dtype)
        else:
            scale_dt = str(scale.dtype).replace("torch.", "")
        if scale_dt in ("uint64", "int64"):
            scale_shape = tuple(scale.shape)
            new_scale = _util.generate_u64_scale(scale_shape)
            if scale_dt == "int64":
                new_scale = new_scale.astype(np.int64)
            _util.write_back(scale, new_scale)


# ============================================================================
# TestSpec classes
# ============================================================================


class AclnnQuantMatmulV3TestSpec:
    """aclnnQuantMatmulV3: (x1, x2, scale, offset, bias, transposeX1, transposeX2, out)

    无 pertoken_scale，group_size=0。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        offset=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        out=None,
        **kwargs,
    ):
        # 1) torch → numpy
        x1_np, x2_np, scale_np, offset_np, bias_np = _util.to_numpy(
            x1, x2, scale, offset, bias
        )

        # 2) UINT64 解码
        deq_scale, u64_offset = _util.decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        # 4) dtype + output shape 推断
        out_dtype = _util.dtype_to_str(out if out is not None else x1.dtype)
        out_shapes = _out_shape(out)

        # 5) 调 kernel golden
        return _kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            offset_np,
            bias_np,
            None,
            transpose_x1=transposeX1,
            transpose_x2=transposeX2,
            group_size=0,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        offset=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        out=None,
        **kwargs,
    ):
        _customize_inputs_impl(x1, x2, scale, offset, bias, None, **kwargs)


class AclnnQuantMatmulV4TestSpec:
    """aclnnQuantMatmulV4: (x1, x2, scale, offset, pertokenScaleOptional, bias, transposeX1, transposeX2, out)

    比 V3 多 pertokenScaleOptional，group_size=0。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        scale,
        offset=None,
        pertokenScaleOptional=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        out=None,
        **kwargs,
    ):
        x1_np, x2_np, scale_np, offset_np, pertoken_np, bias_np = _util.to_numpy(
            x1, x2, scale, offset, pertokenScaleOptional, bias
        )

        deq_scale, u64_offset = _util.decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        out_dtype = _util.dtype_to_str(out if out is not None else x1.dtype)
        out_shapes = _out_shape(out)

        return _kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            offset_np,
            bias_np,
            pertoken_np,
            transpose_x1=transposeX1,
            transpose_x2=transposeX2,
            group_size=0,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        scale,
        offset=None,
        pertokenScaleOptional=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        out=None,
        **kwargs,
    ):
        _customize_inputs_impl(
            x1, x2, scale, offset, bias, pertokenScaleOptional, **kwargs
        )


class AclnnQuantMatmulWeightNzTestSpec:
    """aclnnQuantMatmulWeightNz: (x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2, groupSize, out)

    参数映射: x2Scale→scale, x1Scale→pertoken_scale, x2Offset→offset
    yScale/x1Offset/yOffset 为保留参数，当前不支持。
    """

    @staticmethod
    def golden(
        x1,
        x2,
        x1Scale,
        x2Scale,
        yScale=None,
        x1Offset=None,
        x2Offset=None,
        yOffset=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        groupSize=0,
        out=None,
        **kwargs,
    ):
        x1_np, x2_np, x2Scale_np, x2Offset_np, bias_np, x1Scale_np = _util.to_numpy(
            x1, x2, x2Scale, x2Offset, bias, x1Scale
        )

        formats = kwargs.get("tensor_formats", ())
        is_mx = (
            x2Scale is not None and _util.dtype_to_str(x2Scale.dtype) == "float8_e8m0"
        )
        if is_mx and _scalenz.is_nz(formats, 1):
            x2_np = _scalenz.aclnn_mx_nz_to_nd(x2)
        if is_mx and _scalenz.is_nz(formats, 3):
            x2Scale_np = _scalenz.aclnn_mx_nz_to_nd(x2Scale, scale=True)
            if transposeX2:
                x2Scale_np = np.swapaxes(x2Scale_np, -3, -2)

        deq_scale, u64_offset = _util.decode_u64_scale(x2Scale_np)
        if u64_offset is not None and x2Offset_np is None:
            x2Offset_np = u64_offset

        out_dtype = _util.dtype_to_str(out if out is not None else x1.dtype)
        out_shapes = _out_shape(out)

        return _kernel_qbmm(
            x1_np,
            x2_np,
            deq_scale,
            x2Offset_np,
            bias_np,
            x1Scale_np,
            transpose_x1=transposeX1,
            transpose_x2=transposeX2,
            group_size=groupSize,
            out_dtype=out_dtype,
            out_shapes=out_shapes,
            **kwargs,
        )

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        x1Scale,
        x2Scale,
        yScale=None,
        x1Offset=None,
        x2Offset=None,
        yOffset=None,
        bias=None,
        transposeX1=False,
        transposeX2=False,
        groupSize=0,
        out=None,
        **kwargs,
    ):
        # 对 x2Scale 和 x1Scale 做 E8M0 清洗 + UINT64 生成
        _customize_inputs_impl(x1, x2, x2Scale, x2Offset, bias, x1Scale, **kwargs)
        if (
            x2Scale is not None
            and _util.dtype_to_str(x2Scale.dtype) == "float8_e8m0"
            and _scalenz.is_nz(kwargs.get("tensor_formats", ()), 3)
        ):
            storage = _scalenz.storage_view(x2Scale)
            cleaned = _util.sanitize_e8m0_scale(
                storage, kwargs.get("testcase_name", "unknown")
            )
            if cleaned is not storage:
                _util.write_back(storage, cleaned)


__spec__ = {
    "aclnnQuantMatmulV3": "AclnnQuantMatmulV3TestSpec",
    "aclnnQuantMatmulV4": "AclnnQuantMatmulV4TestSpec",
    "aclnnQuantMatmulWeightNz": "AclnnQuantMatmulWeightNzTestSpec",
}
