#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np


__golden__ = {
    "kernel": {
        "gelu_grad": "gelu_grad_golden"
    }
}

_MIN_FP32 = np.float32(2 ** (-126))
_CSVALUE = np.float32(0.044715)
_SQURT = np.float32(0.7978846)
_CSVALUE_4 = np.float32(0.0535161122)
_CSVALUE_5 = np.float32(0.3989422804)


def _tanh_compute(input_x):
    input_dtype = input_x.dtype
    if input_dtype == "float16":
        input_x = input_x.astype(np.float32)

    input_abs = np.abs(input_x)
    power_val = np.multiply(input_abs, np.float32(-2))
    exp_val_fp32 = np.exp(power_val)

    up_val_tmp = np.multiply(exp_val_fp32, input_x)
    up_val = np.subtract(input_x, up_val_tmp)

    input_x_tmp = np.add(input_abs, _MIN_FP32)
    down_val_tmp = np.add(exp_val_fp32, np.float32(1))
    down_val = np.multiply(down_val_tmp, input_x_tmp)

    res = np.divide(up_val, down_val)
    return res


def _math_four_compute(data_x):
    datax_pow = np.multiply(data_x, data_x)
    datax_pow1 = np.multiply(datax_pow, data_x)
    datax_muls_c = np.multiply(datax_pow1, _CSVALUE)
    datax_addx = np.add(datax_muls_c, data_x)
    datax_muls_s = np.multiply(datax_addx, _SQURT)
    return datax_muls_s


def _result2_compute(data_x):
    data_x_sqr = np.multiply(data_x, data_x)
    data_x_sqr_vmul = np.multiply(data_x_sqr, _CSVALUE_4)
    data_x_sqr_vmul_add1 = np.add(data_x_sqr_vmul, _CSVALUE_5)
    return data_x_sqr_vmul_add1


def _result3_compute(data_x):
    math_four = _math_four_compute(data_x)
    tanh_math_four = _tanh_compute(math_four)
    tanh_math_four_squ = np.multiply(tanh_math_four, tanh_math_four)
    math_four_squ_n = np.multiply(tanh_math_four_squ, np.float32(-1.0))
    add_compute = np.add(math_four_squ_n, np.float32(1.0))
    result3 = np.multiply(add_compute, data_x)
    return result3, tanh_math_four


def _result_grad_compute(data_x):
    result2 = _result2_compute(data_x)
    result3, _ = _result3_compute(data_x)
    mul_result2_3 = np.multiply(result2, result3)
    _, tanh_math_four_result = _result3_compute(data_x)
    mul_compute_1 = np.add(tanh_math_four_result, 1.0)
    mul_compute_2 = np.multiply(mul_compute_1, 0.5)
    res_grad = np.add(mul_compute_2, mul_result2_3)
    return res_grad


def gelu_grad_golden(dy, x, y, **kwargs):
    '''
    Golden function for gelu_grad.
    All the parameters (names and order) follow @gelu_grad_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    '''
    import torch
    from packaging import version

    input_dtype = dy.dtype
    has_improve_precision = False
    
    if version.parse(torch.__version__) >= version.parse("1.12.0"):
        if input_dtype.name in ["float16", "bfloat16"]:
            dy = dy.astype(np.float32)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
        
        dy_torch = torch.from_numpy(dy)
        x_torch = torch.from_numpy(x)
        result = torch.ops.aten.gelu_backward(dy_torch, x_torch, approximate="tanh")
        result = result.numpy().astype(input_dtype, copy=False)
        return result
    else:
        if input_dtype.name == "float16":
            dy = dy.astype(np.float32)
            x = x.astype(np.float32)
            y = y.astype(np.float32)
            has_improve_precision = True

        result5 = _result_grad_compute(x)
        result_temp1 = np.multiply(dy, result5)
        y_temp_1 = np.multiply(y, 0)
        result = np.add(result_temp1, y_temp_1)

        if has_improve_precision:
            result = result.astype(input_dtype, copy=False)
        return result
