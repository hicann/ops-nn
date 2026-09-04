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
flat_quant aclnn golden 实现。

aclnn 接口参数名使用 camelCase（与 C API 一致），golden 委托 kernel golden 的 compute 函数。
"""

import os
import sys


sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)
sys.path.insert(0, os.path.dirname(__file__))

import numeric_ulp  # noqa: E402
import flat_quant_golden as _kernel  # noqa: E402

compute_mx_fp4 = _kernel.compute_mx_fp4
compute_int4 = _kernel.compute_int4
MXFP4_DST_DTYPE = _kernel.MXFP4_DST_DTYPE
numeric_ulp_compare = numeric_ulp.numeric_ulp_compare

_GE_DTYPE_TO_DST = {
    "float4_e2m1": MXFP4_DST_DTYPE,
    "int4": 29,
    "int32": 29,
}


class AclnnFlatQuantTestSpec:
    """aclnnFlatQuant golden。参数名与 C API 一致。"""

    @staticmethod
    def compare(*outputs, **kwargs):
        n = len(outputs) // 2
        results = []
        for i in range(n):
            r = numeric_ulp_compare(outputs[i], outputs[n + i])
            results.append(r)
        return results

    @staticmethod
    def golden(
        x, kroneckerP1, kroneckerP2, clipRatio=1.0, out=None, quantScale=None, **kwargs
    ):
        out_dtype_str = kwargs.get("output_dtypes", [None])[0]
        dst_dtype = _GE_DTYPE_TO_DST.get(
            str(out_dtype_str).split(".")[-1], MXFP4_DST_DTYPE
        )
        if dst_dtype == MXFP4_DST_DTYPE:
            return compute_mx_fp4(x, kroneckerP1, kroneckerP2)
        return compute_int4(x, kroneckerP1, kroneckerP2, clipRatio)


class AclnnFlatQuantV2TestSpec:
    """aclnnFlatQuantV2 golden。额外参数 dstTypeMax。"""

    compare = AclnnFlatQuantTestSpec.compare

    @staticmethod
    def golden(
        x,
        kroneckerP1,
        kroneckerP2,
        clipRatio=1.0,
        dstTypeMax=0.0,
        out=None,
        quantScale=None,
        **kwargs,
    ):
        out_dtype_str = kwargs.get("output_dtypes", [None])[0]
        dst_dtype = _GE_DTYPE_TO_DST.get(
            str(out_dtype_str).split(".")[-1], MXFP4_DST_DTYPE
        )
        if dst_dtype == MXFP4_DST_DTYPE:
            return compute_mx_fp4(x, kroneckerP1, kroneckerP2)
        return compute_int4(x, kroneckerP1, kroneckerP2, clipRatio)


class AclnnFlatQuantV3TestSpec:
    """aclnnFlatQuantV3 golden。额外参数 groupListOptional, groupListType。"""

    compare = AclnnFlatQuantTestSpec.compare

    @staticmethod
    def golden(
        x,
        kroneckerP1,
        kroneckerP2,
        groupListOptional=None,
        clipRatio=1.0,
        dstTypeMax=0.0,
        groupListType=0,
        out=None,
        quantScale=None,
        **kwargs,
    ):
        out_dtype_str = kwargs.get("output_dtypes", [None])[0]
        dst_dtype = _GE_DTYPE_TO_DST.get(
            str(out_dtype_str).split(".")[-1], MXFP4_DST_DTYPE
        )
        if dst_dtype == MXFP4_DST_DTYPE:
            return compute_mx_fp4(x, kroneckerP1, kroneckerP2)
        return compute_int4(x, kroneckerP1, kroneckerP2, clipRatio)


__spec__ = {
    "aclnnFlatQuant": "AclnnFlatQuantTestSpec",
    "aclnnFlatQuantV2": "AclnnFlatQuantV2TestSpec",
    "aclnnFlatQuantV3": "AclnnFlatQuantV3TestSpec",
}
