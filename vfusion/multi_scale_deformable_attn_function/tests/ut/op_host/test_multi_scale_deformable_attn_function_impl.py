#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Dynamic MultiScaleDeformableAttnFunction ut case
"""
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("MultiScaleDeformableAttnFunction", "impl.dynamic.multi_scale_deformable_attn_function", "multi_scale_deformable_attn_function")

ut_case.add_case(["Ascend910A"], {
    "params": [{"shape": (-1, -1, -1, -1), "dtype": "float32",
                "ori_shape": (1, 500, 8, 32),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 10000), (1, 10000), (1, 10000), (1, 10000)]},
               {"shape": (-1, -1), "dtype": "int32",
                "ori_shape": (4, 2),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 8), (1, 8)]},
               {"shape": (-1, -1), "dtype": "int32",
                "ori_shape": (4, 1),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 8), (1, 8)]},
               {"shape": (-1, -1, -1, -1, -1, -1), "dtype": "float32",
                "ori_shape": (1, 5000, 8, 4, 8, 2),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000)]},
               {"shape": (-1, -1, -1, -1, -1), "dtype": "float32",
                "ori_shape": (1, 5000, 8, 4, 8),
                "format": "ND", "ori_format": "ND",
                "range": [(1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000)]},
               {"shape": (-1, -1, -1), "dtype": "float32",
               "ori_shape": (1, 5000, 256),
               "format": "ND", "ori_format": "ND",
                "range": [(1, 10000), (1, 10000), (1, 10000)]}],
    "case_name": "test_success",
    "expect": "success",
    "support_expect": True})
