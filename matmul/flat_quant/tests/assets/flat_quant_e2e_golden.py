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
"""E2E golden specs: torch flat_quant tests delegate to the aclnn golden implementations.

torch 通路输出为 4bit 打包载体 (int32: INT4, uint8: MXFP4), golden 保持 logical dtype,
比对前先将 NPU 输出解包对齐。
"""

import os
import sys

import numpy as np

# 安装 torch 扩展时注册 torch.ops 自定义命名空间; kernel/aclnn 通路加载本资产时无该可选依赖
try:
    import cann_ops_nn  # noqa: F401
except ImportError:
    pass

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
sys.path.insert(0, os.path.dirname(__file__))

import matmul_golden_util as _util  # noqa: E402
import flat_quant_aclnn_golden as _aclnn  # noqa: E402

# torch 侧 dst_dtype 编码: 16(torch.quint4x2) -> INT4 打包为 int32; 296(torch_npu.float4_e2m1fn_x2)
# -> MXFP4 打包为 uint8
_TORCH_QUINT4X2_DTYPE = 16
_TORCH_FLOAT4_E2M1FN_X2_DTYPE = 296


def _as_numpy(value):
    if hasattr(value, "detach"):
        return _util.torch_to_numpy(value)
    return np.asarray(value)


def _dst_is_mxfp4(dst_dtype):
    """兼容 torch dtype 对象与整数编码两种来源."""
    name = str(dst_dtype).split(".")[-1]
    return dst_dtype == _TORCH_FLOAT4_E2M1FN_X2_DTYPE or name in (
        "float4_e2m1fn_x2",
        "float4_e2m1",
    )


def _unpack_fp4x2(packed):
    """uint8 载体解包为 logical fp4_e2m1 视图, 每字节 2 个 fp4, 低半字节在前."""
    codes = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    codes[..., 0::2] = packed & 0x0F
    codes[..., 1::2] = packed >> 4
    return codes.view(_util.np_fp4_e2m1)


def _logical_actual(actual, golden):
    """torch 打包输出按 golden 的 logical dtype 解包: uint8->fp4x2/e8m0, int32->int4x8."""
    actual = _as_numpy(actual)
    golden_dtype = str(golden.dtype)
    if actual.dtype == np.dtype(np.uint8):
        if "float4_e2m1" in golden_dtype:
            return _unpack_fp4x2(actual)
        if "float8_e8m0" in golden_dtype:
            return actual.view(_util.np_fp8_e8m0)
    if actual.dtype == np.dtype(np.int32) and "int4" in golden_dtype:
        return _util.unpack_int32_to_int4(actual)
    return actual


class CannOpsNnFlatQuantTestSpec:
    """cann_ops_nn.flat_quant golden: 委托 aclnn golden 的 Kronecker 分解 + 量化实现."""

    @staticmethod
    def compare(*outputs, **kwargs):
        """多输出精度比对: ttk 传入顺序为 npu0, npu1, ..., golden0, golden1, ..., 按位置配对."""
        compare_context = kwargs.get("compare_context")
        n = len(outputs) // 2
        results = []
        for i in range(n):
            golden = _as_numpy(outputs[n + i])
            actual = _logical_actual(outputs[i], golden)
            results.append(
                _util.numeric_ulp_compare(
                    actual, golden, compare_context=compare_context
                )
            )
        return results

    @staticmethod
    def golden(
        x,
        kronecker_p1,
        kronecker_p2,
        *,
        clip_ratio=1.0,
        dst_dtype=_TORCH_QUINT4X2_DTYPE,
        dst_type_max=0.0,
        group_list=None,
        group_list_type=0,
        **kwargs,
    ):
        x = _as_numpy(x)
        p1 = _as_numpy(kronecker_p1)
        p2 = _as_numpy(kronecker_p2)
        if _dst_is_mxfp4(dst_dtype):
            return _aclnn.compute_mx_fp4(x, p1, p2)
        return _aclnn.compute_int4(x, p1, p2, clip_ratio)


__spec__ = {
    "cann_ops_nn.flat_quant": "CannOpsNnFlatQuantTestSpec",
    "torch.ops.cann_ops_nn.flat_quant": "CannOpsNnFlatQuantTestSpec",
}
