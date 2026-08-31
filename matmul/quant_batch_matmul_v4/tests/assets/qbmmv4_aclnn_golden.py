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
"""ACLNN 层 golden 实现 (仅覆盖 Ascend 950 场景)。

对应 C++ aclnnQuantMatmulV5 接口,按输入参数路由到 V4 或 V3 kernel golden。

ACLNN 层 golden 流程:
  1. torch → numpy
  2. NZ→ND 格式转换 (x2)
  3. INT32→INT4 预处理模拟 (INT4_ASYMMETRICAL 场景)
  4. V4/V3 路由判断
  5. V4 路径: groupSize 解码 + 调 qbmmv4_kernel_compute
     V3 路径: 参数映射 + UINT64 解码 + 调 qbmmv3_kernel_compute
  6. 返回 numpy 结果

参数名严格对齐 C API aclnnQuantMatmulV5GetWorkspaceSize 声明。
"""

import os
import sys
import importlib.util

import numpy as np


# ============================================================================
# Import V4 utilities + kernel golden
# ============================================================================

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../../common/tests/st/arch35"
    ),
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matmul_golden_util import (
    nz_to_nd,
    dtype_to_str,
    torch_to_numpy,
    torch_dtype_to_str,
    write_back,
)
from matmul_quant_util import (
    u64_to_deq_scale,
    u64_to_offset,
    sanitize_e8m0_scale,
    generate_u64_scale,
)
from qbmmv4_kernel_golden import (
    qbmmv4_kernel_compute,
    FP8_INPUT_DTYPE,
    FP4_INPUT_DTYPE,
)

# CANN dtype枚举值映射
_DTYPE_TO_CANN_ENUM = {
    "bfloat16": 27,  # DT_BF16
    "float16": 1,  # DT_FLOAT16
    "float32": 0,  # DT_FLOAT
    "int8": 2,  # DT_INT8
    "int32": 6,  # DT_INT32
    "int64": 7,  # DT_INT64
}

# ============================================================================
# Import V3 kernel golden (isolated to avoid module conflicts)
# ============================================================================

_v3_assets_path = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../quant_batch_matmul_v3/tests/assets",
    )
)
_v3_kernel_golden_path = os.path.join(_v3_assets_path, "qbmmv3_kernel_golden.py")

_saved_sys_path = sys.path.copy()
_saved_sys_modules = {
    k: v
    for k, v in sys.modules.items()
    if k in ("matmul_golden_util", "matmul_quant_util", "qbmmv4_kernel_golden")
}

for mod_name in ("matmul_golden_util", "matmul_quant_util", "qbmmv4_kernel_golden"):
    if mod_name in sys.modules:
        del sys.modules[mod_name]

sys.path = [_v3_assets_path] + [
    p for p in _saved_sys_path if "quant_batch_matmul_v4" not in p
]

_v3_spec = importlib.util.spec_from_file_location(
    "v3_kernel_golden", _v3_kernel_golden_path
)
v3_kernel_golden = importlib.util.module_from_spec(_v3_spec)
_v3_spec.loader.exec_module(v3_kernel_golden)
_qbmmv3_kernel_compute = v3_kernel_golden.qbmmv3_kernel_compute

sys.path = _saved_sys_path
sys.modules.update(_saved_sys_modules)

for mod_name in ("matmul_golden_util", "matmul_quant_util", "qbmmv4_kernel_golden"):
    if mod_name in sys.modules:
        del sys.modules[mod_name]


# ============================================================================
# shared helpers
# ============================================================================


def _to_np(*tensors):
    """torch tensor -> numpy。None 保持 None，已是 numpy 则跳过。"""
    result = []
    for t in tensors:
        if t is None:
            result.append(None)
        elif isinstance(t, np.ndarray):
            result.append(t)
        else:
            result.append(torch_to_numpy(t))
    return tuple(result)


def _out_dtype(out, fallback):
    """返回 out tensor 的 dtype 名，若 out 为 None 则用 fallback。

    对应 C++ GetDtypeAndTranspose: dtype = out->GetDataType()
    """
    if out is not None:
        (out_np,) = _to_np(out)
        if out_np is not None:
            return dtype_to_str(out_np.dtype)
    if isinstance(fallback, np.ndarray):
        return dtype_to_str(fallback.dtype)
    return torch_dtype_to_str(fallback.dtype)


def _out_shape(out):
    """从 out tensor 获取 output shape。"""
    if out is None:
        return ()
    (out_np,) = _to_np(out)
    if out_np is not None:
        return (out_np.shape,)
    return ()


def _nz_to_nd_if_needed(tensor_np, tensor_idx, kwargs):
    """若 tensor 为 FRACTAL_NZ 格式，转换为 ND。"""
    tensor_formats = kwargs.get("tensor_formats", ())
    if len(tensor_formats) <= tensor_idx or tensor_formats[tensor_idx] != "FRACTAL_NZ":
        return tensor_np
    storage_shapes = kwargs.get("tensor_storage_shapes", ())
    ori_shape = storage_shapes[tensor_idx] if len(storage_shapes) > tensor_idx else None
    if ori_shape is not None and tuple(tensor_np.shape) != tuple(ori_shape):
        tensor_np = nz_to_nd(tensor_np, ori_shape)
    return tensor_np


# ============================================================================
# V4/V3 路由判断 (对齐 C++ aclnn_quant_matmul_v5.cpp:1164)
# ============================================================================


def _is_v4_branch(x1_np, x2_np, x1_scale_np, x2_scale_np, x2_offset_np):
    """判断是否走 V4 算子路径。

    对应 C++:
      if (isA8W4 || isA8W8Perblock || isA4W4PergroupNonSymmetric)
          -> l0op::QuantBatchMatmulV4
      else
          -> l0op::QuantBatchMatmulV3
    """
    x1_dt = dtype_to_str(x1_np.dtype)
    x2_dt = dtype_to_str(x2_np.dtype)

    # isA8W4Float: FP8 x FP4 (MX 或 T_CG)
    if x1_dt in FP8_INPUT_DTYPE and x2_dt in FP4_INPUT_DTYPE:
        return True

    # isA8W8Perblock: INT8 x INT8 + FLOAT32 多维 scale (scale.ndim == x.ndim)
    if (
        x1_dt == "int8"
        and x2_dt == "int8"
        and x1_scale_np is not None
        and x2_scale_np is not None
        and dtype_to_str(x1_scale_np.dtype) == "float32"
        and dtype_to_str(x2_scale_np.dtype) == "float32"
        and x1_scale_np.ndim == x1_np.ndim
        and x2_scale_np.ndim == x2_np.ndim
    ):
        return True

    # isA4W4PergroupNonSymmetric: INT4 x INT4 + FP32 2D scale + FP16 2D x2Offset
    if (
        x1_dt == "int4"
        and x2_dt == "int4"
        and x1_scale_np is not None
        and x2_scale_np is not None
        and x2_offset_np is not None
        and dtype_to_str(x1_scale_np.dtype) == "float32"
        and x1_scale_np.ndim >= 2
        and dtype_to_str(x2_scale_np.dtype) == "float32"
        and x2_scale_np.ndim >= 2
        and dtype_to_str(x2_offset_np.dtype) == "float16"
        and x2_offset_np.ndim >= 2
    ):
        return True

    return False


# ============================================================================
# V4 路径: 调 qbmmv4_kernel_compute
# ============================================================================


def _v4_compute(
    x1_np,
    x2_np,
    bias_np,
    x1_scale_np,
    x2_scale_np,
    y_scale_np,
    x2_offset_np,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype,
    **kwargs,
):
    """V4 路径: 调 qbmmv4_kernel_compute。"""
    temp_kwargs = dict(kwargs)
    if out_dtype:
        temp_kwargs["output_dtypes"] = [out_dtype]

    # 将out_dtype字符串转换为CANN枚举值
    dtype_enum = _DTYPE_TO_CANN_ENUM.get(out_dtype, -1)

    result = qbmmv4_kernel_compute(
        x1_np,
        x2_np,
        bias_np,
        x1_scale_np,
        x2_scale_np,
        y_scale_np,
        None,
        x2_offset_np,
        None,
        None,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        dtype=dtype_enum,
        compute_type=-1,
        group_size=group_size,
        **temp_kwargs,
    )

    return result[0]


# ============================================================================
# V3 路径: 参数映射 + UINT64 解码 + 调 qbmmv3_kernel_compute
# ============================================================================


def _decode_u64_scale(scale_np):
    """UINT64/int64 scale 解码为 float32 deq_scale + offset。"""
    if scale_np is None:
        return scale_np, None
    scale_dt = dtype_to_str(scale_np.dtype)
    if scale_dt not in ("uint64", "int64"):
        return scale_np, None
    deq_scale = u64_to_deq_scale(scale_np)
    offset = u64_to_offset(scale_np)
    return deq_scale, offset


def _v3_compute(
    x1_np,
    x2_np,
    bias_np,
    x1_scale_np,
    x2_scale_np,
    x2_offset_np,
    transpose_x1,
    transpose_x2,
    group_size,
    out_dtype,
    out_shapes,
    **kwargs,
):
    """V3 路径: 参数映射 V5->V3 + UINT64 解码 + 调 qbmmv3_kernel_compute。

    参数映射 (对应 aclnn_quant_matmul_v5.cpp:1171-1173):
      x2Scale -> scale, x2Offset -> offset, x1Scale -> pertoken_scale
    """
    scale = x2_scale_np
    offset = x2_offset_np
    pertoken_scale = x1_scale_np

    if scale is not None:
        scale_dt = dtype_to_str(scale.dtype)
        if scale_dt in ("uint64", "int64"):
            deq_scale, u64_offset = _decode_u64_scale(scale)
            scale = deq_scale
            if offset is None:
                offset = u64_offset

    temp_kwargs = dict(kwargs)
    if out_dtype:
        temp_kwargs["output_dtypes"] = [out_dtype]
    if scale is not None:
        temp_kwargs["scale_dtype"] = dtype_to_str(scale.dtype)
    if out_shapes:
        temp_kwargs["output_shapes"] = out_shapes
        temp_kwargs["output_ori_shapes"] = out_shapes

    # 将out_dtype字符串转换为CANN枚举值
    dtype_enum = _DTYPE_TO_CANN_ENUM.get(out_dtype, 0)

    result = _qbmmv3_kernel_compute(
        x1_np,
        x2_np,
        scale,
        offset,
        bias_np,
        pertoken_scale,
        dtype=dtype_enum,
        transpose_x1=transpose_x1,
        transpose_x2=transpose_x2,
        group_size=group_size,
        **temp_kwargs,
    )

    return result[0]


# ============================================================================
# customize_inputs helpers
# ============================================================================


def _customize_inputs_impl(x1, x2, x1Scale, x2Scale, yScale, bias, **kwargs):
    """ACLNN 模式的 customize_inputs 核心逻辑 (in-place 修改)。

    1. E8M0 NaN 清洗 (x1, x2, x1Scale, x2Scale)
    2. UINT64 scale 生成 (x2Scale, yScale)
    """
    testcase_name = kwargs.get("testcase_name", "unknown")

    # 1. E8M0 NaN 清洗
    for tensor in [x1, x2, x1Scale, x2Scale]:
        if tensor is None:
            continue
        cleaned = sanitize_e8m0_scale(tensor, testcase_name)
        if cleaned is not tensor:
            write_back(tensor, cleaned)

    # 2. UINT64 scale 生成
    for tensor in [x2Scale, yScale]:
        if tensor is None:
            continue
        if isinstance(tensor, np.ndarray):
            tensor_dt = dtype_to_str(tensor.dtype)
        else:
            tensor_dt = str(tensor.dtype).replace("torch.", "")
        if tensor_dt in ("uint64", "int64"):
            tensor_shape = tuple(tensor.shape)
            new_scale = generate_u64_scale(tensor_shape)
            if tensor_dt == "int64":
                new_scale = new_scale.astype(np.int64)
            write_back(tensor, new_scale)


# ============================================================================
# TestSpec
# ============================================================================


class AclnnQuantMatmulV5TestSpec:
    """aclnnQuantMatmulV5 的 TestSpec 类。

    参数顺序对齐 C API:
      aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, yScale,
        x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2,
        groupSize, out, workspaceSize, executor)
    """

    @staticmethod
    def golden(
        x1,
        x2,
        x1Scale=None,
        x2Scale=None,
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
        """ACLNN golden: torch->numpy + NZ->ND + V4/V3 路由 + kernel 计算。"""
        (
            x1_np,
            x2_np,
            x1_scale_np,
            x2_scale_np,
            y_scale_np,
            x2_offset_np,
            bias_np,
        ) = _to_np(x1, x2, x1Scale, x2Scale, yScale, x2Offset, bias)

        x2_np = _nz_to_nd_if_needed(x2_np, 1, kwargs)

        out_dtype = _out_dtype(out, x1)
        out_shapes = _out_shape(out)

        is_v4 = _is_v4_branch(x1_np, x2_np, x1_scale_np, x2_scale_np, x2_offset_np)

        if is_v4:
            result = _v4_compute(
                x1_np,
                x2_np,
                bias_np,
                x1_scale_np,
                x2_scale_np,
                y_scale_np,
                x2_offset_np,
                transposeX1,
                transposeX2,
                groupSize,
                out_dtype,
                **kwargs,
            )
        else:
            result = _v3_compute(
                x1_np,
                x2_np,
                bias_np,
                x1_scale_np,
                x2_scale_np,
                x2_offset_np,
                transposeX1,
                transposeX2,
                groupSize,
                out_dtype,
                out_shapes,
                **kwargs,
            )

        return result

    @staticmethod
    def customize_inputs(
        x1,
        x2,
        x1Scale=None,
        x2Scale=None,
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
        """输入预处理: E8M0 NaN 清洗 + UINT64 scale 生成。"""
        _customize_inputs_impl(x1, x2, x1Scale, x2Scale, yScale, bias, **kwargs)


__spec__ = {
    "aclnnQuantMatmulV5": "AclnnQuantMatmulV5TestSpec",
}
