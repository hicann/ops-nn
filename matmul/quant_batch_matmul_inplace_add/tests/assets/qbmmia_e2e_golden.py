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

继承 ACLNN golden，对应 torch API: torch_npu.npu_add_quant_matmul_
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../../common/tests/st/arch35")
)

import en_dtypes
import matmul_golden_util as _util
from qbmmia_aclnn_golden import AclnnQuantBatchMatmulInplaceAddTestSpec


def _group_sizes_to_int(group_sizes):
    if group_sizes is None:
        return 0
    if isinstance(group_sizes, (list, tuple)):
        gs_m = int(group_sizes[0]) if len(group_sizes) > 0 else 0
        gs_n = int(group_sizes[1]) if len(group_sizes) > 1 else 0
        gs_k = int(group_sizes[2]) if len(group_sizes) > 2 else 0
        return (gs_m << 32) | (gs_n << 16) | gs_k
    return int(group_sizes)


class TorchNpuAddQuantMatmulTestSpec(AclnnQuantBatchMatmulInplaceAddTestSpec):
    @classmethod
    def golden(
        cls,
        self,
        x1,
        x2,
        x2_scale,
        *,
        x1_scale=None,
        group_sizes=None,
        x1_dtype=None,
        x2_dtype=None,
        x1_scale_dtype=None,
        x2_scale_dtype=None,
        **kwargs,
    ):
        transpose_x1 = _util.detect_transpose_from_strides(x1)
        transpose_x2 = _util.detect_transpose_from_strides(x2)

        group_size = _group_sizes_to_int(group_sizes)

        # hif8 场景：检测非连续布局，恢复存储布局后转为 hifloat8 numpy
        if x1_dtype is not None and int(x1_dtype) == 290:
            if isinstance(x1, torch.Tensor) and not x1.is_contiguous():
                x1 = np.ascontiguousarray(_util.torch_to_numpy(x1).T).view(
                    en_dtypes.hifloat8
                )
                transpose_x1 = True
            else:
                x1 = _util.torch_to_numpy(x1).view(en_dtypes.hifloat8)
        if x2_dtype is not None and int(x2_dtype) == 290:
            if isinstance(x2, torch.Tensor) and not x2.is_contiguous():
                x2 = np.ascontiguousarray(_util.torch_to_numpy(x2).T).view(
                    en_dtypes.hifloat8
                )
                transpose_x2 = True
            else:
                x2 = _util.torch_to_numpy(x2).view(en_dtypes.hifloat8)

        # mxfp8 场景：将 int8 scale 转回 float8_e8m0 numpy
        if (
            x1_scale_dtype is not None
            and int(x1_scale_dtype) == 293
            and x1_scale is not None
        ):
            x1_scale = _util.torch_to_numpy(x1_scale).view(en_dtypes.float8_e8m0)
        if (
            x2_scale_dtype is not None
            and int(x2_scale_dtype) == 293
            and x2_scale is not None
        ):
            x2_scale = _util.torch_to_numpy(x2_scale).view(en_dtypes.float8_e8m0)

        return super().golden(
            x1,
            x2,
            x1_scale,
            x2_scale,
            self,
            transposeX1=transpose_x1,
            transposeX2=transpose_x2,
            groupSize=group_size,
            **kwargs,
        )

    @classmethod
    def customize_inputs(
        cls,
        self,
        x1,
        x2,
        x2_scale,
        *,
        x1_scale=None,
        group_sizes=None,
        x1_dtype=None,
        x2_dtype=None,
        x1_scale_dtype=None,
        x2_scale_dtype=None,
        **kwargs,
    ):
        super().customize_inputs(x1, x2, x1_scale, x2_scale, self, **kwargs)

        # hif8 场景：生成 hifloat8 数据 -> view(uint8) -> write_back
        if x1_dtype is not None and int(x1_dtype) == 290:
            x1_data = (
                np.random.uniform(-1, 1, x1.shape)
                .astype(np.float32)
                .astype(en_dtypes.hifloat8)
            )
            _util.write_back(x1, x1_data.view(np.uint8))

        if x2_dtype is not None and int(x2_dtype) == 290:
            x2_data = (
                np.random.uniform(-1, 1, x2.shape)
                .astype(np.float32)
                .astype(en_dtypes.hifloat8)
            )
            _util.write_back(x2, x2_data.view(np.uint8))

        # mxfp8 场景：生成 float8_e8m0 scale 数据 -> view(int8) -> write_back
        if (
            x1_scale_dtype is not None
            and int(x1_scale_dtype) == 293
            and x1_scale is not None
        ):
            x1_scale_data = (
                np.random.uniform(0, 5, x1_scale.shape)
                .astype(np.float32)
                .astype(en_dtypes.float8_e8m0)
            )
            _util.write_back(x1_scale, x1_scale_data.view(np.int8))

        if (
            x2_scale_dtype is not None
            and int(x2_scale_dtype) == 293
            and x2_scale is not None
        ):
            x2_scale_data = (
                np.random.uniform(0, 5, x2_scale.shape)
                .astype(np.float32)
                .astype(en_dtypes.float8_e8m0)
            )
            _util.write_back(x2_scale, x2_scale_data.view(np.int8))


__spec__ = {
    "torch_npu.npu_add_quant_matmul_": "TorchNpuAddQuantMatmulTestSpec",
}
