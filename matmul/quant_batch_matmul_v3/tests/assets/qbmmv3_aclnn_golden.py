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
import matmul_quant_util as _quant
import qbmmv3_kernel_golden as _kernel


# ============================================================================
# shared helpers
# ============================================================================


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
    """返回 out tensor 的 dtype 名，若 out 为 None 则用 fallback 的 dtype。

    对应 C++ GetDtypeAndTranspose (aclnn_quant_matmul_v4.cpp:1633):
      dtype = static_cast<int64_t>(out->GetDataType());
    """
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return _util.dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return _util.dtype_to_str(fallback.dtype)
    return _util.torch_dtype_to_str(fallback.dtype)


def _out_shape(out):
    """从 out tensor 获取 output shape（用于 determine_quant_mode 推断 M/N）。"""
    if out is None:
        return ()
    (out_np,) = _to_np(out)
    if out_np is not None:
        return (out_np.shape,)
    return ()


def _nz_to_nd_if_needed(tensor_np, tensor_idx, kwargs):
    """若 tensor_formats 指示 tensor 为 FRACTAL_NZ，转换为 ND。

    ACLNN 层在此完成转换，传给 kernel golden 的 tensor 始终为 ND。
    """
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= tensor_idx or tensor_formats[tensor_idx] != "FRACTAL_NZ":
        return tensor_np
    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[tensor_idx] if len(storage_shapes) > tensor_idx else None
    if ori_shape is not None and tuple(tensor_np.shape) != tuple(ori_shape):
        tensor_np = _util.nz_to_nd(tensor_np, ori_shape)
    return tensor_np


def _decode_u64_scale(scale_np):
    """UINT64/int64 scale 解码为 float32 deq_scale + offset。

    对应 kernel golden golden() 方法中的 customize_inputs 逻辑。
    """
    if scale_np is None:
        return scale_np, None
    scale_dt = _util.dtype_to_str(scale_np.dtype)
    if scale_dt not in ("uint64", "int64"):
        return scale_np, None
    deq_scale = _quant.u64_to_deq_scale(scale_np)
    offset = _quant.u64_to_offset(scale_np)
    return deq_scale, offset


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
    if scale_np is not None:
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
    input_ranges = kwargs.get("input_ranges", None)

    # 1. E8M0 NaN 清洗（in-place write_back）
    for idx, tensor in enumerate([x1, x2, scale, None, None, pertoken_scale]):
        if tensor is None:
            continue
        cleaned = _quant.sanitize_e8m0_scale(tensor, idx, input_ranges, testcase_name)
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
            new_scale = _quant.generate_u64_scale(scale_shape)
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
        x1_np, x2_np, scale_np, offset_np, bias_np = _to_np(x1, x2, scale, offset, bias)

        # 2) NZ→ND
        x2_np = _nz_to_nd_if_needed(x2_np, 1, kwargs)
        scale_np = _nz_to_nd_if_needed(scale_np, 2, kwargs)

        # 3) UINT64 解码
        deq_scale, u64_offset = _decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        # 4) dtype + output shape 推断
        out_dtype = _out_dtype(out, x1)
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
        x1_np, x2_np, scale_np, offset_np, pertoken_np, bias_np = _to_np(
            x1, x2, scale, offset, pertokenScaleOptional, bias
        )

        x2_np = _nz_to_nd_if_needed(x2_np, 1, kwargs)
        scale_np = _nz_to_nd_if_needed(scale_np, 2, kwargs)

        deq_scale, u64_offset = _decode_u64_scale(scale_np)
        if u64_offset is not None and offset_np is None:
            offset_np = u64_offset

        out_dtype = _out_dtype(out, x1)
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
        x1_np, x2_np, x2Scale_np, x2Offset_np, bias_np, x1Scale_np = _to_np(
            x1, x2, x2Scale, x2Offset, bias, x1Scale
        )

        # x2 为 NZ 格式
        x2_np = _nz_to_nd_if_needed(x2_np, 1, kwargs)

        deq_scale, u64_offset = _decode_u64_scale(x2Scale_np)
        if u64_offset is not None and x2Offset_np is None:
            x2Offset_np = u64_offset

        out_dtype = _out_dtype(out, x1)
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


__spec__ = {
    "aclnnQuantMatmulV3": "AclnnQuantMatmulV3TestSpec",
    "aclnnQuantMatmulV4": "AclnnQuantMatmulV4TestSpec",
    "aclnnQuantMatmulWeightNz": "AclnnQuantMatmulWeightNzTestSpec",
}
