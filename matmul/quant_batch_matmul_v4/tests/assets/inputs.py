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
__input__ = {
        "kernel": {
            "quant_batch_matmul_v4": "quant_batch_matmul_v4_inputs"
        }
}

import numpy as np

def quant_batch_matmul_v4_inputs(x1, x2, bias = None, x1_scale = None, x2_scale = None, y_scale = None,
                                 x1_offset = None, x2_offset = None, y_offset = None, x2_table = None,
                                 dtype:int = -1, compute_type:int = -1,
                                 transpose_x1:bool = False, transpose_x2:bool = False,
                                 group_size:int = -1, **kwargs):
    testcase_name = kwargs['testcase_name']
    # convert scale to uint64
    if (y_scale is not None) and (y_scale.dtype == "uint64"):
        y_scale_shape = y_scale.shape
        y_scale = afp8wfp4_scale_generate(y_scale_shape, testcase_name)
    if (x2_scale is not None) and (x2_scale.dtype == "uint64"):
        x2_scale = u64_scale_generate(x2_scale.shape, testcase_name)

    return x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, x2_table

def afp8wfp4_scale_generate(y_scale_shape, testcase_name):
    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=y_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(y_scale_shape)
    # 与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000

    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(y_scale_shape)
    np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
    uint64_deq_scale = np.zeros(y_scale_shape, np.uint64)
    uint64_deq_scale |= np.uint64(uint32_deq_scale)

    return uint64_deq_scale

def u64_scale_generate(scale_shape, testcase_name):

    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(scale_shape)
    # 与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000

    fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(scale_shape)
    np.save(testcase_name + "_x2_scale.npy", fp32_deq_scale)
    uint64_deq_scale = np.zeros(scale_shape, np.uint64)
    uint64_deq_scale |= np.uint64(uint32_deq_scale)

    return uint64_deq_scale
