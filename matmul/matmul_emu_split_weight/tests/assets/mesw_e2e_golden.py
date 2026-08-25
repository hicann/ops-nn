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

委托 mesw_aclnn_golden 完成计算。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import mesw_aclnn_golden as _aclnn


class TorchMatmulEmuSplitWeightTestSpec:
    @staticmethod
    def golden(
        x, w_high, w_low, *, w_low_scale=0.00390625, y_dtype=0, out=None, **kwargs
    ):
        """torch.matmul_emu_split_weight."""
        return _aclnn.AclnnMatmulEmuSplitWeightTestSpec.golden(
            x,
            w_high,
            w_low,
            out=out,
            w_low_scale=w_low_scale,
            y_dtype=y_dtype,
            **kwargs,
        )


__spec__ = {
    "torch.matmul_emu_split_weight": "TorchMatmulEmuSplitWeightTestSpec",
}
