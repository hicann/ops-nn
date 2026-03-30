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
            "quant_batch_matmul_v3": "quant_batch_matmul_v3_inputs"
        }
}

import numpy as np

def quant_batch_matmul_v3_inputs(x1, x2, scale, offset = None, bias = None, pertoken_scale = None, *, dtype: int,
                                 transpose_x1: bool = False, transpose_x2: bool = False,
                                 group_size:int = 0, **kwargs):
    # 获取数据
    input_deq_scale = scale
    output_dtypes = kwargs['output_dtypes']
    out_dtype = output_dtypes[0]
    testcase_name = kwargs['testcase_name']

    # convert scale to uint64
    if input_deq_scale.dtype == "uint64" and pertoken_scale is None:
        deq_scale_shape = scale.shape
        input_deq_scale = scale_generate(deq_scale_shape, offset, out_dtype, testcase_name)

    return x1, x2, input_deq_scale, offset, bias, pertoken_scale

def scale_generate(deq_scale_shape, offset, out_dtype, testcase_name):
    has_offset = offset is not None

    fp32_deq_scale = np.random.uniform(low=-5, high=5, size=deq_scale_shape).astype(np.float32)
    uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
    # 与高19位运算，模拟硬件
    uint32_deq_scale &= 0XFFFFE000

    if has_offset:
        offset_shape = offset.shape
        fp32_offset = np.random.uniform(low=-5, high=5, size=offset_shape).astype(np.float32)

    # dequant
    if out_dtype != "int8":
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
    # requant
    elif out_dtype == "int8":
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32).reshape(deq_scale_shape)
        np.save(testcase_name + "_deq_scale.npy", fp32_deq_scale)
        s9_offset = 0
        if has_offset:
            np.save(testcase_name + "_offset.npy", fp32_offset)
            s9_offset = f32_2_s9(fp32_offset).astype(int).reshape(offset_shape)
            s9_offset &= 0X1FF
            s9_offset = s9_offset[0] if deq_scale_shape[-1] < offset_shape[-1] else s9_offset
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)
        uint64_deq_scale |= np.uint64(s9_offset << 37)
        uint64_deq_scale |= 1 << 46
    return uint64_deq_scale

def f32_2_s9(array):
    array_round = np.round(array)
    array_round_clip = np.clip(array_round, -256, 255)
    return array_round_clip