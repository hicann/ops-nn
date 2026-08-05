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
"""
weight_quant_batch_matmul_v2 ACLNN golden implementation.

V2/Nz 和 V3 拆分为两个 class，因为 V3 API 比 V2 多一个 innerPrecise 位置参数。
"""

import os
import sys
import numpy as np
from importlib import import_module

_assets = os.path.dirname(os.path.abspath(__file__))
_common = os.path.join(_assets, "../../../common/tests/st/arch35")

for _d in (_common, _assets):
    if _d not in sys.path:
        sys.path.insert(0, _d)

_util = import_module("matmul_golden_util")
_quant_util = import_module("matmul_quant_util")
_kernel = import_module("wqbmmv2_kernel_golden")


def _common_golden(
    x,
    weight,
    antiquantScale,
    antiquantOffsetOptional,
    quantScaleOptional,
    quantOffsetOptional,
    biasOptional,
    antiquantGroupSize,
    y,
    kwargs,
):
    """V2/V3/Nz 公共 golden 逻辑：torch输入转换 → 调用kernel计算 → 返回结果。"""
    if antiquantGroupSize is None:
        antiquantGroupSize = kwargs.get("antiquant_group_size", 0)

    transpose_x = kwargs.get("transpose_x", False) or kwargs.get("transposeX", False)
    transpose_weight = kwargs.get("transpose_weight", False) or kwargs.get(
        "transposeWeight", False
    )
    if not transpose_weight:
        transpose_weight = _util.detect_transpose_from_strides(weight)
    if not transpose_x:
        transpose_x = _util.detect_transpose_from_strides(x)

    tensor_formats = kwargs.get("tensor_formats", None)
    if tensor_formats and len(tensor_formats) >= 7:
        input_formats = tuple(tensor_formats[:7])
    else:
        input_formats = ("ND",) * 7

    np_args = [
        _util.torch_to_numpy(t)
        for t in (
            x,
            weight,
            antiquantScale,
            antiquantOffsetOptional,
            quantScaleOptional,
            quantOffsetOptional,
            biasOptional,
        )
    ]

    tensor_dtypes = kwargs.get("tensor_dtypes", None)
    weight_dtype_str = (
        tensor_dtypes[1] if tensor_dtypes and len(tensor_dtypes) > 1 else None
    )

    _preprocess_nz_weight(np_args, input_formats, weight_dtype_str)
    _preprocess_scale_dtype(np_args, tensor_dtypes, kwargs)
    _unpack_packed_weight(np_args, weight_dtype_str, transpose_weight)

    input_ori_shapes_out = tuple(a.shape if a is not None else None for a in np_args)
    out_dtype_str = _util.torch_dtype_to_str(y.dtype)

    result = _kernel.wqbmm_kernel_compute(
        *np_args,
        antiquant_group_size=antiquantGroupSize,
        transpose_x=transpose_x,
        transpose_weight=False,
        output_dtypes=[out_dtype_str],
        input_formats=input_formats,
        input_ori_shapes=input_ori_shapes_out,
        **kwargs,
    )
    return result[0]


def _preprocess_nz_weight(np_args, input_formats, weight_dtype_str):
    weight_format = input_formats[1] if len(input_formats) > 1 else "ND"
    if (
        weight_format == "FRACTAL_NZ"
        and np_args[1] is not None
        and np_args[1].ndim == 2
    ):
        np_args[1] = _util.nz_2d_to_nd(np_args[1], weight_dtype_str)


def _preprocess_scale_dtype(np_args, tensor_dtypes, kwargs):
    scale_dtype_str = (
        tensor_dtypes[2] if tensor_dtypes and len(tensor_dtypes) > 2 else None
    )
    if scale_dtype_str == "float8_e8m0" and np_args[2] is not None:
        np_args[2] = np_args[2].astype(_quant_util.np_mx_scale)
        np_args[2] = _quant_util.sanitize_e8m0_scale(
            np_args[2],
            2,
            kwargs.get("input_ranges", None),
            kwargs.get("testcase_name", "unknown"),
        )


def _unpack_packed_weight(np_args, weight_dtype_str, transpose_weight):
    if np_args[1] is None:
        return
    if weight_dtype_str == "int32":
        if transpose_weight:
            np_args[1] = np.ascontiguousarray(np_args[1].T)
        np_args[1] = _quant_util.unpack_int32_to_int4(np_args[1], axis=-1)
        if transpose_weight:
            np_args[1] = np.swapaxes(np_args[1], -2, -1)
    elif weight_dtype_str == "float32":
        if transpose_weight:
            np_args[1] = np.ascontiguousarray(np_args[1].T)
        np_args[1] = _quant_util.unpack_float_to_fp4(
            np_args[1].astype(np.float32), axis=-1
        )
        if transpose_weight:
            np_args[1] = np.swapaxes(np_args[1], -2, -1)


def _aclnn_customize_inputs(
    x,
    weight,
    antiquantScale,
    antiquantOffsetOptional,
    quantScaleOptional,
    quantOffsetOptional,
    biasOptional,
    antiquantGroupSize=None,
    **kwargs,
):
    """ACLNN customize_inputs：e8m0 NaN 修复 + 950 约束校验，in-place 写回。

    ACLNN 路径框架不捕获 customize_inputs 返回值，必须 in-place 修改。
    """
    input_ranges = kwargs.get("input_ranges", None)
    testcase_name = kwargs.get("testcase_name", "unknown")

    if antiquantScale is not None:
        sanitized = _quant_util.sanitize_e8m0_scale(
            antiquantScale, 2, input_ranges, testcase_name
        )
        if sanitized is not antiquantScale:
            _util.write_back(antiquantScale, sanitized)

    _quant_util.validate_wqbmmv2_constraints(
        x,
        weight,
        antiquantScale,
        antiquantOffsetOptional,
        quantScaleOptional,
        quantOffsetOptional,
        biasOptional,
        **kwargs,
    )


class AclnnWeightQuantBatchMatmulV2TestSpec:
    """ACLNN golden for V2/Nz: antiquantGroupSize 后直接是 y。"""

    @staticmethod
    def golden(
        x,
        weight,
        antiquantScale,
        antiquantOffsetOptional,
        quantScaleOptional,
        quantOffsetOptional,
        biasOptional,
        antiquantGroupSize=None,
        y=None,
        **kwargs,
    ):
        return _common_golden(
            x,
            weight,
            antiquantScale,
            antiquantOffsetOptional,
            quantScaleOptional,
            quantOffsetOptional,
            biasOptional,
            antiquantGroupSize,
            y,
            kwargs,
        )

    def customize_inputs(
        x,
        weight,
        antiquantScale,
        antiquantOffsetOptional,
        quantScaleOptional,
        quantOffsetOptional,
        biasOptional,
        antiquantGroupSize=None,
        y=None,
        **kwargs,
    ):
        _aclnn_customize_inputs(
            x,
            weight,
            antiquantScale,
            antiquantOffsetOptional,
            quantScaleOptional,
            quantOffsetOptional,
            biasOptional,
            antiquantGroupSize,
            **kwargs,
        )


class AclnnWeightQuantBatchMatmulV3TestSpec:
    """ACLNN golden for V3: antiquantGroupSize 后是 innerPrecise，再是 y。"""

    @staticmethod
    def golden(
        x,
        weight,
        antiquantScale,
        antiquantOffsetOptional,
        quantScaleOptional,
        quantOffsetOptional,
        biasOptional,
        antiquantGroupSize=None,
        innerPrecise=None,
        y=None,
        **kwargs,
    ):
        return _common_golden(
            x,
            weight,
            antiquantScale,
            antiquantOffsetOptional,
            quantScaleOptional,
            quantOffsetOptional,
            biasOptional,
            antiquantGroupSize,
            y,
            kwargs,
        )

    def customize_inputs(
        x,
        weight,
        antiquantScale,
        antiquantOffsetOptional,
        quantScaleOptional,
        quantOffsetOptional,
        biasOptional,
        antiquantGroupSize=None,
        innerPrecise=None,
        y=None,
        **kwargs,
    ):
        _aclnn_customize_inputs(
            x,
            weight,
            antiquantScale,
            antiquantOffsetOptional,
            quantScaleOptional,
            quantOffsetOptional,
            biasOptional,
            antiquantGroupSize,
            **kwargs,
        )


__spec__ = {
    "aclnnWeightQuantBatchMatmulV2": "AclnnWeightQuantBatchMatmulV2TestSpec",
    "aclnnWeightQuantBatchMatmulV3": "AclnnWeightQuantBatchMatmulV3TestSpec",
    "aclnnWeightQuantBatchMatmulNz": "AclnnWeightQuantBatchMatmulV2TestSpec",
}
